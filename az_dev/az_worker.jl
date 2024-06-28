struct MDPAgent{M <: MDP, MCTS <: GumbelSearch, RNG <: AbstractRNG, H <: History}
    mdp     :: M
    mcts    :: MCTS
    rng     :: RNG
    history :: H
end

function MDPAgent(mdp::MDP; rng::AbstractRNG = Random.default_rng(), kwargs...)
    mcts    = GumbelSearch(mdp; rng, kwargs...)
    state   = rand(rng, initialstate(mdp))
    history = History(mdp, state)
    return MDPAgent(mdp, mcts, rng, history)
end

function initialize_agent!(agent::MDPAgent)
    (; mcts, history) = agent
    insert_root!(mcts, history.current_state)
    s_querry = mcts_forward!(mcts)
    return s_querry
end


struct Worker{AC, A <: MDPAgent, B <: BatchManager, H <: History}
    agents          :: Vector{A}
    batch_manager   :: B
    history_channel :: Channel{H}
    actor_critic    :: AC
    batchsize       :: Int
end

function Worker(; mdp::MDP, actor_critic, n_agents::Int, batchsize::Int=n_agents, kwargs...)
    @assert n_agents >= batchsize >= Threads.nthreads() - 2

    agents = [MDPAgent(mdp; kwargs...) for _ in 1:n_agents]

    n_batches = ceil(Int, 1 + n_agents / batchsize)
    in_size   = size(rand(MersenneTwister(1), initialstate(mdp)))
    na        = length(actions(mdp))
    batch_manager = BatchManager(n_batches, batchsize, in_size, na)

    history_channel = Channel{History{statetype(mdp), Float32}}(n_agents)

    for (agent_idx, agent) in enumerate(agents)
        s_querry = initialize_agent!(agent)
        set_querry!(batch_manager, agent_idx, s_querry)
    end

    actor_critic = gpu(actor_critic)

    return Worker(agents, batch_manager, history_channel, actor_critic, batchsize)
end

function worker_main(worker::Worker; n_steps::Int = 10_000)
    stop_flag = Threads.Atomic{Bool}(false)

    @sync begin
        errormonitor(Threads.@spawn actor_critic_worker(worker, stop_flag))

        # dont start workers until the first response is ready, otherwise wont work
        wait(worker.batch_manager.batches[1].response_ready)

        errormonitor(Threads.@spawn begin
            querries_per_step = length(worker.agents) * (1 + worker.agents[1].mcts.tree_querries)

            counter = 0
            steps = 0

            while !stop_flag[]
                if !response_ready(worker.batch_manager)
                    GC.safepoint()
                    continue
                end

                @sync for _ in 1:worker.batch_manager.batchsize
                    Threads.@spawn process_agent(worker)
                end

                counter += worker.batch_manager.batchsize
                steps += worker.batch_manager.batchsize
                if counter >= querries_per_step
                    counter -= querries_per_step
                    println("querries: $steps")
                    GC.gc(false)
                end
            end
        end)

        episode_returns, histories = fetch(Threads.@spawn history_worker(worker, n_steps))

        Threads.atomic_xchg!(stop_flag, true)

        return episode_returns, histories
    end
end

function history_worker(worker::Worker{AC, A, B, H}, n_steps::Int) where {AC, A, B, H}
    steps_taken = 0
    episode_returns = Float64[]
    histories = H[]
    while steps_taken < n_steps
        hist = take!(worker.history_channel)
        steps_taken += length(hist.state)
        push!(episode_returns, hist.episode_reward)
        push!(histories, hist)
        # println("steps taken: $steps_taken")
    end
    return episode_returns, histories
end

function actor_critic_worker(worker::Worker, stop_flag::Threads.Atomic{Bool})
    while !stop_flag[]
        if querry_ready(worker.batch_manager)
            process_batch!(worker.batch_manager, worker.actor_critic)
        end
        GC.safepoint()
    end
    return nothing
end

function agent_worker(worker::Worker, stop_flag::Threads.Atomic{Bool})
    while !stop_flag[]
        if response_ready(worker.batch_manager)
            process_agent(worker)
        end
        GC.safepoint()
    end
    return nothing
end

function process_agent(worker::Worker)
    (; batch_manager, history_channel) = worker

    agent_idx, value, policy = get_response!(batch_manager)
    agent = worker.agents[agent_idx]

    mcts_backward!(agent.mcts, value, policy)

    if Base.isdone(agent.mcts)
        step_agent!(agent, history_channel)
    end

    s_querry = mcts_forward!(agent.mcts)
    set_querry!(batch_manager, agent_idx, s_querry)

    return nothing
end

function step_agent!(agent::MDPAgent, history_channel::Channel)
    (; mdp, mcts, rng, history) = agent

    a, a_info = root_info(mcts)
    sp, reward = @gen(:sp, :r)(mdp, history.current_state, a, rng)

    push!(history;
        sp,
        reward,
        policy_target = a_info.policy_target,
        value_target  = a_info.value_target
    )

    if isterminal(mdp, sp) || length(history.state) >= 250
        calculate_value_target(history)
        newstate = rand(rng, initialstate(mdp))
        history_copy = copy_and_reset!(history, newstate)
        put!(history_channel, history_copy)
    end

    insert_root!(mcts, history.current_state)

    return nothing
end
