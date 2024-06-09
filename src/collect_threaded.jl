struct GPUQuerry{C<:Channel, QC<:Array{Float32}, QG<:CuArray, V<:Vector, P<:Matrix}
    channels::Vector{C} 
    cpu_batched_querry::QC
    gpu_batched_querry::QG
    cpu_value::V
    cpu_policy::P
end

function GPUQuerry(; sz::Tuple, na::Int, batchsize::Int)
    return GPUQuerry(
        Channel[], 
        zeros(Float32, sz..., batchsize),
        CUDA.zeros(Float32, sz..., batchsize),
        zeros(Float32, batchsize),
        zeros(Float32, na, batchsize)
    )
end

function getpolicyvalue_gpu(querry_channel::Channel, response_channel::Channel)
    function getpolicyvalue(b)
        b_rep = input_representation(b)
        put!(querry_channel, (response_channel, b_rep))
        out = take!(response_channel)
        return (; value=out.value[], policy=vec(out.policy))
    end
    return getpolicyvalue
end

function gen_data_threaded(pomdp::POMDP, net, params::minBetaZeroParameters, buffer::DataBuffer)
    (; n_episodes, GumbelSolver_args, na, input_dims) = params
    
    storage_channel = Channel{Any}()
    data_task = Threads.@spawn collect_data(buffer, storage_channel, n_episodes)
    errormonitor(data_task)

    @sync begin
        querry_ch = Channel(n_episodes)

        worker_tasks = [threaded_worker(querry_ch, storage_channel, pomdp, params) for _ in 1:n_episodes]
        
        gpu_compute_ch = Channel{GPUQuerry}(2)

        loading_channel = Channel{GPUQuerry}(2; spawn=true) do loading_ch
            for Q in loading_ch
                load_querries(Q, querry_ch, gpu_compute_ch)
            end
        end
    
        errormonitor(Threads.@spawn gpu_compute(gpu_compute_ch, loading_channel, net))

        for _ in 1:2
            Q = GPUQuerry(; sz = (input_dims..., GumbelSolver_args.n_particles), na, batchsize = 16)
            put!(loading_channel, Q)
        end

        wait.(worker_tasks)

        # need to close more gracefully, can still throw errors like this
        close(querry_ch)
        close(loading_channel)
        close(gpu_compute_ch)
    end

    return fetch(data_task)
end

function threaded_worker(querry_ch::Channel, storage_channel::Channel, pomdp::POMDP, params::minBetaZeroParameters)
    (; GumbelSolver_args) = params

    worker_ch = Channel()

    worker_task = Threads.@spawn begin
        getpolicyvalue = getpolicyvalue_gpu(querry_ch, worker_ch)

        solver = GumbelSolver(; getpolicyvalue, GumbelSolver_args...)
        planner = solve(solver, pomdp)

        data = work_fun(pomdp, planner, params)
        put!(storage_channel, data)

        return nothing
    end

    errormonitor(worker_task)
    bind(worker_ch, worker_task)

    return worker_task
end

function load_querries(Q::GPUQuerry, querry_ch::Channel, gpu_compute_ch::Channel)
    lock(querry_ch) do
        isready(querry_ch) || wait(querry_ch)

        empty!(Q.channels)

        for cpu_querry in eachslice(Q.cpu_batched_querry; dims=ndims(Q.cpu_batched_querry))
            isready(querry_ch) || break
            (ch, querry) = take!(querry_ch)
            push!(Q.channels, ch)
            copyto!(cpu_querry, querry)
        end

        put!(gpu_compute_ch, Q)
    end
end

function gpu_compute(gpu_compute_ch::Channel, loading_channel::Channel, net)
    net = gpu(net)
    for Q in gpu_compute_ch
        Threads.@spawn begin
            copyto!(Q.gpu_batched_querry, Q.cpu_batched_querry)
            net_out = net(Q.gpu_batched_querry)

            copyto!(Q.cpu_value, net_out.value)
            copyto!(Q.cpu_policy, net_out.policy)

            CUDA.synchronize()

            for (i, ch) in enumerate(Q.channels)
                response = (value = Q.cpu_value[i], policy = Q.cpu_policy[:,i])
                put!(ch, response)
            end

            empty!(Q.channels)

            lock(loading_channel) do
                isopen(loading_channel) && put!(loading_channel, Q)
            end
        end
    end
    return nothing
end