module minBetaZero

using Flux, CUDA
using POMDPs, POMDPTools
using ParticleFilters
using Statistics, Distributions, Random
using Pkg
using ProgressMeter, Plots
using Distributed

include("ParticleFilterTrees/ParticleFilterTrees.jl")
using .ParticleFilterTrees
export PFTBelief, GuidedTree, GumbelSolver, GumbelPlanner

include("neural_network.jl")
using .NeuralNet
export NetworkParameters, ActorCritic, getloss

export betazero, minBetaZeroParameters, input_representation

function input_representation end

include("tools.jl")

@kwdef struct minBetaZeroParameters
    GumbelSolver_args = (;
        max_depth           = 10,
        n_particles         = 100,
        tree_queries        = 100,
        max_time            = Inf,
        k_o                 = 24.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 1_000, 
        beliefcache_size    = 1_000,
        cscale              = 1.,
        cvisit              = 50.,
        stochastic_root     = true,
        m_acts_init
    )
    t_max       = 100
    n_episodes  = 500
    n_iter      = 20
    batchsize   = 128
    lr          = 3e-4
    lambda      = 0.0
    plot_training = false
    train_device  = cpu
    inference_device = cpu
    input_dims = (1,)
    na = 3
    buff_cap = 10_000
    n_particles = 500
    use_belief_reward = true
    use_gumbel_target = true
    train_intensity = 4
    warmup_steps = buff_cap ÷ train_intensity
end

struct RandPlanner{A} <: Policy
    actions::A
end
RandPlanner(p::POMDP) = RandPlanner(actions(p))
POMDPs.action(p::RandPlanner, b) = rand(p.actions)

struct netPolicy{N,A} <: Policy
    net::N
    ordered_actions::A
end
POMDPs.action(p::netPolicy, b) = p.ordered_actions[argmax(getlogits(p.net, b))]

struct netPolicyStoch{N,A} <: Policy
    net::N
    ordered_actions::A
end
POMDPs.action(p::netPolicyStoch, b) = p.ordered_actions[argmax(getlogits(p.net, b) + rand(Gumbel(), length(p.ordered_actions)))]

getlogits(net::ActorCritic, b) = vec(net(input_representation(b); logits=true).policy)

function betazero(params::minBetaZeroParameters, pomdp::POMDP, net)
    (; n_iter, input_dims, na, buff_cap, n_particles, train_intensity, warmup_steps) = params

    @assert nworkers() > 1 || Threads.nthreads() > 1 """
    Error: Distributed computing is not available. 
    Please run `addprocs()` or start Julia with `--threads` to a value greater than 1.
    """

    info = Dict(
        :steps => Int[],
        :episodes => Int[],
        :returns => Float64[],
        :returns_error => Float64[],
        :network_returns => Float64[],
        :network_returns_error => Float64[],
        :training => Dict[]
    )

    na = length(actions(pomdp))

    buffer = DataBuffer((input_dims..., n_particles), na, buff_cap)

    for itr in 1:n_iter+1
        iter_timer = time()

        @info itr <= n_iter ? "Gathering Data - Iteration: $itr" : "Gathering Data - Final Evaluation"

        if nworkers() > 1 # prioritize distributed ?
            returns, steps = gen_data_distributed(pomdp, net, params, buffer)
        else
            returns, steps = gen_data_threaded(pomdp, net, params, buffer)
        end
        
                
        episodes = length(returns)
        returns_mean = mean(returns)
        returns_error = std(returns)/sqrt(length(returns))
        push!.(
            (info[:steps], info[:episodes], info[:returns], info[:returns_error]), 
            (steps, episodes, returns_mean, returns_error)
        )
        _rm = round(returns_mean; sigdigits=1+Base.hidigit(returns_mean,10)-Base.hidigit(returns_error,10))
        _re = round(returns_error; sigdigits=1)
        @info "Mean return $_rm +/- $_re"
        @info "Gathered $steps data points over $episodes episodes"

        net_results = test_network(net, pomdp, params; n_episodes=100)
        returns_mean = mean(net_results)
        returns_error = std(net_results)/sqrt(length(net_results))
        push!.(
            (info[:network_returns], info[:network_returns_error]),
            (returns_mean, returns_error)
        )
        _rm = round(returns_mean; sigdigits=1+Base.hidigit(returns_mean,10)-Base.hidigit(returns_error,10))
        _re = round(returns_error; sigdigits=1)
        @info "Network mean return $_rm +/- $_re"

        # This is so wez get the stats after the n_iter network update
        itr == n_iter+1 && break

        if buffer.length >= warmup_steps
            idxs = rand(1:buffer.length, steps*train_intensity)
            train_data = (; 
                x             = select_last_dim(buffer.network_input, idxs),
                value_target  = select_last_dim(buffer.value_target , idxs),
                policy_target = select_last_dim(buffer.policy_target, idxs)
            )
            net, train_info = train!(net, train_data, params)
            push!(info[:training], train_info)
        end

        iter_time = ceil(Int, time()-iter_timer)
        @info "Iteration time: $iter_time seconds"
        println()
    end

    info[:steps] = cumsum(info[:steps])
    info[:episodes] = cumsum(info[:episodes])

    return net, info
end

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

function collect_data(buffer::DataBuffer, storage_channel, n_episodes::Int)
    progress = Progress(n_episodes)
    ret_vec = zeros(Float32, n_episodes)
    steps = 0

    for i in 1:n_episodes
        data = take!(storage_channel)
        steps += length(data.value_target)
        set_buffer!(
            buffer; 
            network_input = data.network_input, 
            value_target  = data.value_target, 
            policy_target = data.policy_target
        )
        ret_vec[i] = data.returns
        next!(progress)
    end
    
    close(storage_channel)

    return ret_vec, steps
end

function getpolicyvalue_cpu(net)
    function getpolicyvalue(b)
        b_rep = input_representation(b)
        out = net(b_rep; logits=true)
        return (; value=out.value[], policy=vec(out.policy))
    end
    return getpolicyvalue
end

function getpolicyvalue_gpu(querry_channel, response_channel)
    function getpolicyvalue(b)
        b_rep = input_representation(b)
        put!(querry_channel, (response_channel, b_rep))
        out = take!(response_channel)
        return (; value=out.value[], policy=vec(out.policy))
    end
    return getpolicyvalue
end

@kwdef struct GPUQuerry{C<:Channel, A<:AbstractArray, L<:Base.AbstractLock, E<:Threads.Event}
    channels::Vector{C} = Channel[]
    querries::A
    lock::L = ReentrantLock()
    hasdata::E = Event()
end
GPUQuerry(sz) = GPUQuerry(; querries = CUDA.pin(zeros(Float32, sz)))

function togpuchannel!(ch::Channel, batched_querries::GPUQuerry)
    batched_data = (copy(batched_querries.channels), cu(batched_querries.querries))
    put!(ch, batched_data)
    empty!(batched_querries.channels)
    reset(batched_querries.hasdata)
end


function gen_data_threaded(pomdp::POMDP, net, params::minBetaZeroParameters, buffer::DataBuffer)
    (; n_episodes, inference_device, GumbelSolver_args) = params

    storage_channel  = Channel{Any}(n_episodes)
    data_task = Threads.@spawn collect_data(buffer, storage_channel, n_episodes)
    errormonitor(data_task)
    bind(storage_channel, data_task)

    @sync begin

        gpu_results_ch = Channel{Tuple{<:Any, <:Any}}(n_episodes; spawn=true) do ch
            for (response_chs, gpu_results) in ch
                cpu_results = cpu(gpu_results)
                for (i, response_ch) in enumerate(response_chs)
                    response = (; value=cpu_results.value[i], policy=cpu_results.policy[:,i])
                    put!(response_ch, response)
                end
            end
            return nothing
        end

        gpu_compute_ch = Channel{Tuple{Vector{<:Channel}, <:CuArray}}(n_episodes, spawn=true) do ch
            gpu_net = gpu(net)
            for (response_chs, input) in ch
                gpu_results = gpu_net(input; logits=true)
                put!(gpu_results_ch, (response_chs, gpu_results))
            end
            return nothing
        end

        querry_ch = Channel{Tuple{<:Channel, <:Array}}(n_episodes)

        worker_responses = [
            Channel{@NamedTuple{value::Float32, policy::Vector{Float32}}}(; spawn=true) do response_ch
                getpolicyvalue = getpolicyvalue_gpu(querry_ch, response_ch)
                solver = GumbelSolver(; getpolicyvalue, GumbelSolver_args...)
                planner = solve(solver, pomdp)
                data = work_fun(pomdp, planner, params)
                put!(storage_channel, data)
                return nothing
            end
            for _ in 1:n_episodes
        ]

        batchsize = 64
        arr_sz = (size(buffer.network_input)[1:end-1]..., batchsize)
        aggregator(gpu_compute_ch, querry_ch, worker_responses, arr_sz)

        wait(data_task)
        close(gpu_results_ch)
        close(gpu_compute_ch)
        close(querry_ch)
    end

    ret_vec, steps = fetch(data_task)

    return ret_vec, steps
end




function aggregator(
    gpu_compute_ch::Channel{<:Tuple{Vector{<:Channel}, <:AbstractArray}}, 
    querry_ch::Channel{Tuple{<:Channel, <:Array}}, 
    worker_responses::Vector{<:Channel}, 
    arr_dims::Tuple{Vararg{Int}}
    )

    try 
        batched_querries = GPUQuerry(arr_dims)

        Threads.@spawn for (ch, querry) in querry_ch
            lock(batched_querries.lock) do
                push!(batched_querries.channels, ch)
                N = length(batched_querries.channels)
                dst = select_last_dim(batched_querries.querries, N)
                copyto!(dst, querry)

                batchsize = min(arr_dims[end], count(isopen, worker_responses))
                if N == batchsize
                    togpuchannel!(gpu_compute_ch, batched_querries)
                else
                    notify(event)
                end    

                return nothing
            end
        end

        Threads.@spawn begin
            while count(isopen, worker_responses) > 0
                if Base.n_avail(gpu_compute_ch) == 0
                    wait(batched_querries.hasdata)
                    lock(batched_querries.lock) do
                        if length(batched_querries.channels) > 0
                            togpuchannel!(gpu_compute_ch, batched_querries)
                        end
                        return nothing
                    end
                end
            end
        end
    catch e
        if isopen(querry_ch) && isopen(gpu_compute_ch)
            close(querry_ch)
            close(gpu_compute_ch)
            rethrow(e)
        end
    end

    return nothing
end


function work_fun(pomdp, planner, params)
    (; t_max, n_particles, use_belief_reward, use_gumbel_target) = params

    use_gumbel_target = use_gumbel_target && isa(planner, GumbelPlanner)

    up = BootstrapFilter(pomdp, n_particles)
    b = initialize_belief(up, initialstate(pomdp))
    s = rand(initialstate(pomdp))

    b_vec = []
    aid_vec = Int[]
    gumbel_target_vec = Vector[]
    state_reward = Float32[]
    belief_reward = Float32[]

    for _ in 1:t_max
        a, a_info = action_info(planner, b)
        aid = actionindex(pomdp, a)
        s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)

        push!.((b_vec, aid_vec, state_reward), (b, aid, r))

        if use_gumbel_target
            push!(gumbel_target_vec, a_info.policy_target)
        end

        if use_belief_reward
            br = 0f0
            for p in particles(b)
                br += reward(pomdp, p, a) / n_particles
            end
            push!(belief_reward, br)
        end
        
        if isterminal(pomdp, s)
            break
        end

        b = POMDPs.update(up, b, a, o)
    end

    gamma = discount(pomdp)

    for i in length(state_reward)-1:-1:1
        state_reward[i] += gamma * state_reward[i+1]
    end

    if use_belief_reward
        for i in length(belief_reward)-1:-1:1
            belief_reward[i] += gamma * belief_reward[i+1]
        end
    end

    output_reward = use_belief_reward ? belief_reward : state_reward

    if use_gumbel_target
        policy_target = reduce(hcat, gumbel_target_vec)
    else
        policy_target = Flux.onehotbatch(aid_vec, 1:length(actions(pomdp)))
    end

    data = (; 
        network_input = stack(input_representation, b_vec), 
        value_target  = reshape(output_reward, 1, :), 
        policy_target = policy_target, 
        returns       = state_reward[1]
    )

    return data
end

function test_network(net, pomdp, params; n_episodes=500, type=netPolicyStoch)
    ret_vec = pmap(1:n_episodes) do _
        planner = type(net, actions(pomdp))
        data = work_fun(pomdp, planner, params)
        return data.returns
    end
    return ret_vec
end


function train!(net, data, params::minBetaZeroParameters)
    (; batchsize, lr, lambda, train_device, plot_training) = params

    Etrain = mean(-sum(x->iszero(x) ? x : x*log(x), data.policy_target; dims=1))
    varVtrain = var(data.value_target)

    train_batchsize = min(batchsize, Flux.numobs(data))
    train_data = Flux.DataLoader(data; batchsize=train_batchsize, shuffle=true, partial=true) |> train_device

    @info "Training on $(Flux.numobs(data)) samples"

    net = train_device(net)
    opt = Flux.setup(Flux.Optimiser(Flux.Adam(lr), WeightDecay(lr*lambda)), net)

    info = Dict(
        :policy_loss => Float32[], 
        :policy_KL   => Float32[], 
        :value_loss  => Float32[], 
        :value_FVU   => Float32[]
    )
    
    Flux.trainmode!(net) 
    @showprogress for (; x, value_target, policy_target) in train_data
        grads = Flux.gradient(net) do net
            losses = getloss(net, x; value_target, policy_target)

            Flux.Zygote.ignore_derivatives() do 
                push!(info[:policy_loss], losses.policy_loss          )
                push!(info[:policy_KL]  , losses.policy_loss - Etrain )
                push!(info[:value_loss] , losses.value_loss           )
                push!(info[:value_FVU]  , losses.value_mse / varVtrain)
                return nothing
            end

            return losses.value_loss + losses.policy_loss
        end
        Flux.update!(opt, net, grads[1])
    end
    Flux.testmode!(net)

    if plot_training
        plotargs = (; label=false)
        plot(
            plot(info[:value_loss]; ylabel="Value Loss", plotargs...),
            plot(info[:policy_loss]; ylabel="Policy Loss", plotargs...),
            plot(info[:value_FVU]; ylabel="FVU", plotargs...),
            plot(info[:policy_KL]; ylabel="Policy KL", plotargs...)
            ; 
            layout=(2,2), 
            size=(900,600)
        ) |> display
    end

    return cpu(net), info
end




end # module minBetaZero




#= Notes

For all of the data in the buffer, update

trains on the data "reuse" times

gather data
states = 0
for trajectory in generator()
    add to buffer
    states += new_states
    if states > buff_cap / reuse
        break
    end
end




=#