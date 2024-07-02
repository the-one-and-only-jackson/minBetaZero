struct MDPWorker{AC, A <: MDPAgent, B <: BatchManager, H <: Union{<:Channel, <:RemoteChannel}}
    actor_critic    :: AC
    agents          :: Vector{A}
    step_counter    :: Threads.Atomic{Int}
    steps_since_gc  :: Threads.Atomic{Int}
    batch_manager   :: B
    history_channel :: H
end

function MDPWorker(params::AlphaZeroParams, mdp::MDP, actor_critic, history_channel)
    (; max_steps, segment_length, inference_batchsize, tree_queries, k_o, cscale,
    cvisit, m_acts_init, rng, n_agents, inference_T) = params

    worker_args = (; inference_batchsize, n_agents, inference_T, history_channel)
    agent_args  = (; max_steps, segment_length, rng)
    mcts_args   = (; tree_queries, m_acts_init, k_o, cscale, cvisit)

    MDPWorker(; actor_critic, mdp, worker_args..., agent_args..., mcts_args...)
end

function MDPWorker(;
    actor_critic,
    mdp                 :: MDP,
    history_channel     :: Channel,
    inference_batchsize :: Integer = 32,
    n_agents            :: Integer = 2 * actor_critic,
    inference_T         :: Union{Type{Float32}, Type{Float16}} = Float32,
    agentargs...
    )

    batchsize = inference_batchsize # used to match az params name

    @assert n_agents >= batchsize

    agents = [MDPAgent(mdp; agentargs...) for _ in 1:n_agents]

    step_counter   = Threads.Atomic{Int}(0)
    steps_since_gc = Threads.Atomic{Int}(0)

    in_size       = size(rand(MersenneTwister(1), initialstate(mdp)))
    na            = length(actions(mdp))
    n_batches     = ceil(Int, n_agents // batchsize)
    batch_manager = BatchManager{inference_T}(; batchsize, in_size, na, n_batches)

    # initialize all the agents
    for (agent_idx, agent) in enumerate(agents)
        s_querry = initialize_agent!(agent)
        set_querry!(batch_manager, agent_idx, s_querry)
    end
    @assert isready(batch_manager)

    actor_critic = actor_critic |> (inference_T === Float16 ? Flux.f16 : Flux.f32) |> gpu

    return MDPWorker(
        actor_critic,
        agents,
        step_counter,
        steps_since_gc,
        batch_manager,
        history_channel
    )
end

function update_actor_critic!(worker::MDPWorker, actor_critic)
    foreach(copyto!, Flux.params(worker.actor_critic), Flux.params(actor_critic))
end

function worker_main(worker::MDPWorker, n_steps::Integer; ntasks = Threads.nthreads() - 1)
    (; batch_manager, step_counter) = worker

    buff_len = length(batch_manager.batches) * batch_manager.batchsize
    ch_eltype = Tuple{eltype(batch_manager), Int}

    response_ch = Channel{ch_eltype}(buff_len; spawn=true) do ch
        actor_critic_worker(worker, ch, n_steps)
    end

    Threads.foreach(response_ch; ntasks) do (batch, index)
        process_agent(worker, batch, index)
    end

    # Setting step_counter to 0 vs subtracting n_steps
    # I think subtracting n_steps will lead to more uniform workloads, but not really sure
    Threads.atomic_sub!(step_counter, n_steps)

    return nothing
end

function actor_critic_worker(worker::MDPWorker, response_ch::Channel, target_steps::Integer)
    (; batch_manager, actor_critic, step_counter, steps_since_gc, agents) = worker

    n_agents = length(agents)

    while step_counter[] < target_steps
        if steps_since_gc[] >= n_agents
            Threads.atomic_sub!(steps_since_gc, n_agents) # Think setting to 0 is wrong
            GC.gc(false)
        end

        GC.safepoint() # task is always live, need to insert GC point
        isready(batch_manager) || continue

        batch = take!(batch_manager)
        process_batch!(batch, actor_critic)
        lock(response_ch) do
            foreach(i -> put!(response_ch, (batch, i)), 1:batch.batchsize)
        end
    end

    return nothing
end

function process_agent(worker::MDPWorker, batch::ACBatch, index::Integer)
    (; agents, history_channel, step_counter, steps_since_gc, batch_manager) = worker

    agent_idx, value, policy = get_response!(batch, index)
    agent = agents[agent_idx]
    mcts  = agent.mcts

    mcts_backward!(mcts, value, policy)

    free_querry!(batch_manager, batch, index)

    if Base.isdone(mcts)
        step_agent!(agent, history_channel)
        Threads.atomic_add!(step_counter, 1)
        Threads.atomic_add!(steps_since_gc, 1)
    end

    s_querry = mcts_forward!(mcts)

    set_querry!(batch_manager, agent_idx, s_querry)

    return nothing
end
