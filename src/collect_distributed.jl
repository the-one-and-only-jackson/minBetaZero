function gen_data_distributed(pomdp::POMDP, net, params::minBetaZeroParameters, buffer::DataBuffer)
    (; n_episodes, GumbelSolver_args) = params
    
    storage_channel = RemoteChannel(()->Channel{Any}())

    data_task = Threads.@spawn collect_data(buffer, storage_channel, n_episodes)
    errormonitor(data_task)

    pmap(1:nworkers()) do worker_idx
        getpolicyvalue = getpolicyvalue_cpu(net)
        solver = GumbelSolver(; getpolicyvalue, GumbelSolver_args...)
        planner = solve(solver, pomdp)

        if worker_idx <= mod(n_episodes, nworkers())
            n_episodes = n_episodes ÷ nworkers() + 1
        else
            n_episodes = n_episodes ÷ nworkers()
        end

        for _ in 1:n_episodes
            data = work_fun(pomdp, planner, params)
            put!(storage_channel, data)
        end

        return nothing
    end

    return fetch(data_task)
end

function getpolicyvalue_cpu(net)
    function getpolicyvalue(b)
        b_rep = input_representation(b)
        out = net(b_rep; logits=true)
        return (; value=out.value[], policy=vec(out.policy))
    end
    return getpolicyvalue
end