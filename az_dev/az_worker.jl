struct Worker{AC, A <: MDPAgent, B <: BatchManager, H <: History}
    agents          :: Vector{A}
    batch_manager   :: B
    history_channel :: Channel{H}
    actor_critic    :: AC
    batchsize       :: Int
end

function Worker(; mdp::MDP, actor_critic, n_agents::Int, batchsize::Int=n_agents, kwargs...)
    @assert n_agents >= batchsize

    agents = [MDPAgent(mdp; kwargs...) for _ in 1:n_agents]

    n_batches = ceil(Int, 1 + n_agents / batchsize)
    in_size   = size(rand(MersenneTwister(1), initialstate(mdp)))
    na        = length(actions(mdp))
    batch_manager = BatchManager(n_batches, batchsize, in_size, na)

    H = History{statetype(mdp), Vector{Float32}, Float32}
    history_channel = Channel{H}(n_agents)

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

        for _ in 1:Threads.nthreads() - 2
            errormonitor(Threads.@spawn agent_worker(worker, stop_flag))
        end

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
        println("steps taken: $steps_taken")
    end
    return episode_returns, histories
end

function actor_critic_worker(worker::Worker, stop_flag::Threads.Atomic{Bool})
    while !stop_flag[]
        if querry_ready(worker.batch_manager)
            process_batch!(worker.batch_manager, worker.actor_critic)
        end
        GC.safepoint()
        yield()
    end
    println("AC exited gracefully")
    return nothing
end

function agent_worker(worker::Worker, stop_flag::Threads.Atomic{Bool})
    while !stop_flag[]
        if response_ready(worker.batch_manager)
            process_agent(worker)
        end
        GC.safepoint()
        yield()
    end
    println("Worker exited gracefully")
    return nothing
end

function process_agent(worker::Worker)
    (; batch_manager, history_channel) = worker

    agent_idx, value, policy = get_response!(batch_manager)
    agent = worker.agents[agent_idx]
    s_querry, history = agent_main!(agent, value, policy)
    set_querry!(batch_manager, agent_idx, s_querry)

    isnothing(history) || put!(history_channel, history)

    return
end
