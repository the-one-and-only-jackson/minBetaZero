module minBetaZero

using Flux, CUDA
using POMDPs, POMDPModelTools, POMDPTools
using ParticleFilters
using Statistics, Distributions, Random
using Pkg
using ProgressMeter, Plots

# include("../lib/ParticleFilterTrees/src/ParticleFilterTrees.jl")
using ParticleFilterTrees

include("beliefmdp.jl")
export ParticleBeliefMDP

include("neural_network.jl")
using .NeuralNet
export NetworkParameters, ActorCritic, getloss

# include("pft_interface.jl")
# export NetworkWrapper, PUCT, get_value, get_policy, input_representation
export input_representation
function input_representation end


export betazero, minBetaZeroParameters

function setup()
    Pkg.develop(PackageSpec(url=joinpath(@__DIR__,"..","lib","ParticleFilterTrees")))
end

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
    train_frac  = 0.8
    batchsize   = 128
    lr          = 3e-4
    lambda      = 0.0
    n_epochs    = 50
    plot_training = false
    train_dev   = cpu
    early_stop  = 10
    n_warmup    = 500
end

mutable struct DataBuffer{A,B,C}
    const network_input::A
    const value_target::B
    const policy_target::C
    const capacity::Int
    length::Int
    idx::Int
end

function DataBuffer(input_dims::Tuple{Vararg{Int}}, na::Int, capacity::Int)
    network_input = zeros(Float32, input_dims..., capacity)
    value_target  = zeros(Float32, 1, capacity)
    policy_target = zeros(Float32, na, capacity)
    return DataBuffer(network_input, value_target, policy_target, capacity, 0, 1)
end

function set_buffer!(b::DataBuffer; network_input, value_target, policy_target)
    L = length(value_target)

    @assert size(network_input)[end] == L || (L==1 && ndims(network_input) == ndims(b.network_input) - 1)
    @assert size(value_target )[end] == L || (L==1 && ndims(value_target ) == ndims(b.value_target ) - 1) "value $(size(value_target))"
    @assert size(policy_target)[end] == L || (L==1 && ndims(policy_target) == ndims(b.policy_target) - 1)

    if b.capacity - b.idx + 1 >= L
        dst_idxs = b.idx .+ (0:L-1)
        copyto!(select_last_dim(b.network_input, dst_idxs), network_input)
        copyto!(select_last_dim(b.value_target, dst_idxs), value_target)
        copyto!(select_last_dim(b.policy_target, dst_idxs), policy_target)
    else
        L1 = b.capacity - b.idx + 1
        dst_idxs = b.idx:b.capacity
        src_idxs = 1:L1
        copyto!(select_last_dim(b.network_input, dst_idxs), select_last_dim(network_input, src_idxs))
        copyto!(select_last_dim(b.value_target, dst_idxs), select_last_dim(value_target, src_idxs))
        copyto!(select_last_dim(b.policy_target, dst_idxs), select_last_dim(policy_target, src_idxs))

        L2 = L - L1
        dst_idxs = 1:L2
        src_idxs = L1+1:L
        copyto!(select_last_dim(b.network_input, dst_idxs), select_last_dim(network_input, src_idxs))
        copyto!(select_last_dim(b.value_target, dst_idxs), select_last_dim(value_target, src_idxs))
        copyto!(select_last_dim(b.policy_target, dst_idxs), select_last_dim(policy_target, src_idxs))
    end
    
    b.idx = mod1(b.idx + L, b.capacity)
    b.length = min(b.length + L, b.capacity)

    return nothing
end

@inline select_last_dim(arr, idxs) = selectdim(arr, ndims(arr), idxs)

function gen_data(pomdp, net, params::minBetaZeroParameters, buffer::DataBuffer)
    (; n_episodes) = params

    belief = Vector{Array{Float32}}(undef, n_episodes)
    value = Vector{Vector{Float32}}(undef, n_episodes)
    policy = Vector{Vector{Int}}(undef, n_episodes)
    returns = Vector{Float32}(undef, n_episodes)

    progress = Progress(n_episodes)
    progress_channel = Channel{Bool}(1)

    worker_querries = [Channel{Any}(1) for _ in 1:n_episodes]
    master_responses = [Channel{Any}(1) for _ in 1:n_episodes]

    @sync begin
        async_loop = Threads.@spawn while take!(progress_channel)
            next!(progress)
        end

        Threads.@spawn master_function(net, async_loop, worker_querries, master_responses)

        thread_vec = Vector{Any}(undef, n_episodes)
        for i in 1:n_episodes 
            thread_vec[i] = Threads.@spawn begin
                _data = work_fun(pomdp, params, worker_querries[i], master_responses[i])
                belief[i] = _data.belief
                value[i] = _data.value
                policy[i] = _data.policy
                returns[i] = _data.returns
                put!(progress_channel, true)
            end
        end

        wait.(thread_vec)
        put!(progress_channel, false)
    end

    for (network_input, value_target, ai) in zip(belief, value, policy)
        policy_target = Flux.onehotbatch(ai, 1:length(actions(pomdp)))
        value_target = reshape(value_target,1,:)
        set_buffer!(buffer; network_input, value_target, policy_target)
    end

    steps = sum(length(v) for v in value)

    return returns, steps
end

function master_function(net, async_loop, worker_querries, master_responses)
    master_net = net |> gpu
    while !istaskdone(async_loop) && !istaskfailed(async_loop)
        idxs = isready.(worker_querries)
        iszero(count(idxs)) && continue

        if count(idxs) == 1
            idx = findfirst(idxs)
            querries = take!(worker_querries[idx])
            querries_batched = reshape(querries, size(querries)..., 1) |> gpu
            results = master_net(querries_batched; logits=true) |> cpu
            put!(master_responses[idx], results)    
        else
            querries = take!.(worker_querries[idxs])
            querries_batched = querries |> stack |> gpu
            results = master_net(querries_batched; logits=true) |> cpu
            dims = size(first(results),ndims(first(results)))
            split_results = [(; value=results.value[:,i], policy=results.policy[:,i]) for i in 1:dims]
            put!.(master_responses[idxs], split_results)
        end
    end
end

function work_fun(pomdp, params, querry_channel, response_channel)
    (; t_max, GumbelSolver_args) = params

    function getpolicyvalue(b)
        b_rep = input_representation(b)
        put!(querry_channel, b_rep)
        out = take!(response_channel)
        return (; value=out.value[], policy=vec(out.policy))
    end

    planner = solve(GumbelSolver(; getpolicyvalue, GumbelSolver_args...), pomdp)

    up = BootstrapFilter(pomdp, 500)
    b = initialize_belief(up, initialstate(pomdp))
    s = rand(initialstate(pomdp))

    b_vec = typeof(b)[]
    aid_vec = Int[]
    r_vec = Float32[]
    for _ in 1:t_max
        a = action(planner, b)
        aid = actionindex(pomdp,a)
        s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)
        push!.((b_vec,aid_vec,r_vec), (b,aid,r))
        b = POMDPs.update(up, b, a, o)
        if isterminal(pomdp, s)
            break
        end
    end

    gamma = discount(pomdp)
    for i in length(r_vec)-1:-1:1
        r_vec[i] += gamma * r_vec[i+1]
    end

    episode_return = r_vec[1]

    return (; 
        belief=stack(input_representation, b_vec), 
        value=r_vec, 
        policy=aid_vec, 
        returns=episode_return
    )
end

function train!(net, data, params::minBetaZeroParameters)
    (; train_frac, batchsize, lr, lambda, n_epochs, train_dev, early_stop, n_warmup) = params

    split_data = Flux.splitobs(data, at=train_frac, shuffle=true) 

    Etrain = mean(-sum(x->iszero(x) ? x : x*log(x), split_data[1].policy_target; dims=1))
    Evalid = mean(-sum(x->iszero(x) ? x : x*log(x), split_data[2].policy_target; dims=1))
    varVtrain = var(split_data[1].value_target)
    varVvalid = var(split_data[2].value_target)

    split_data = split_data
    train_data = Flux.DataLoader(split_data[1]; batchsize=min(batchsize, Flux.numobs(split_data[1])), shuffle=true, partial=false) |> train_dev
    valid_data = Flux.DataLoader(split_data[2]; batchsize=min(4*batchsize, Flux.numobs(split_data[2])), shuffle=true, partial=false) |> train_dev

    @info "Training on $(Flux.numobs(split_data[1])) samples"
    @info "Testing on  $(Flux.numobs(split_data[2])) samples"

    net = train_dev(net)
    opt = Flux.setup(Flux.Optimiser(Flux.Adam(lr), WeightDecay(lr*lambda)), net)

    checkpoint = deepcopy(net)
    checkpoint_val = Inf32
    checkpoint_pol = Inf32
    last_checkpoint = 0

    train_info = Dict(:policy_loss=>Float32[], :policy_KL=>Float32[], :value_loss=>Float32[], :value_FVU=>Float32[])
    valid_info = Dict(:policy_loss=>Float32[], :policy_KL=>Float32[], :value_loss=>Float32[], :value_FVU=>Float32[])

    function lossfun(net, x; value_target, policy_target, info, policy_entropy, value_var)
        losses = getloss(net, x; value_target, policy_target)
        @assert all(isfinite, losses.policy_loss)
        @assert all(isfinite, losses.value_loss)
        @assert all(isfinite, losses.value_mse)
        Flux.Zygote.ignore_derivatives() do 
            push!(info[:policy_loss], losses.policy_loss                 )
            push!(info[:policy_KL]  , losses.policy_loss - policy_entropy)
            push!(info[:value_loss] , losses.value_loss                  )
            push!(info[:value_FVU]  , losses.value_mse / value_var       )
        end
        losses.value_loss + losses.policy_loss
    end
    
    epochs_completed = 0
    grad_steps = 0
    @showprogress for epoch in 1:n_epochs
        epochs_completed += 1
        Flux.trainmode!(net)
        for (; x, value_target, policy_target) in train_data
            grad_steps += 1

            if n_epochs*length(train_data)+1 > n_warmup > 0
                ramp = grad_steps / n_warmup
                decay = 1 - (grad_steps - n_warmup) / (n_epochs*length(train_data) - n_warmup)
                lr_scale = min(ramp, decay)
                Flux.adjust!(opt; eta=lr_scale*lr, lambda=lr_scale*lr*lambda)
            end
    
            grads = Flux.gradient(net) do net
                lossfun(net, x; value_target, policy_target, info=train_info, policy_entropy=Etrain, value_var=varVtrain)
            end
            Flux.update!(opt, net, grads[1])
        end

        Flux.testmode!(net)
        for (; x, value_target, policy_target) in valid_data    
            lossfun(net, x; value_target, policy_target, info=valid_info, policy_entropy=Evalid, value_var=varVvalid)
        end

        epoch_pol = mean(valid_info[:policy_loss][end-length(valid_data)+1:end])
        epoch_val = mean(valid_info[:value_loss][end-length(valid_data)+1:end])
        if epoch_pol < checkpoint_pol && epoch_val < checkpoint_val
            checkpoint = deepcopy(net)
            checkpoint_val = epoch_val
            checkpoint_pol = epoch_pol   
            last_checkpoint = epoch  
        end
        if epoch - last_checkpoint >= early_stop
            break
        end
    end

    info = Dict(
        :value_train_loss  => train_info[:value_loss],
        :value_valid_loss  => valid_info[:value_loss],
        :policy_train_loss => train_info[:policy_loss],
        :policy_valid_loss => valid_info[:policy_loss],
        :train_R           => train_info[:value_FVU],
        :valid_R           => valid_info[:value_FVU],
        :train_KL          => train_info[:policy_KL],
        :valid_KL          => valid_info[:policy_KL]
    )

    for (k,v) in info
        info[k] = dropdims(mean(reshape(v,:,epochs_completed); dims=1); dims=1)
    end

    return cpu(checkpoint), info
end

function betazero(params::minBetaZeroParameters, pomdp::POMDP, net)
    (; n_iter, plot_training) = params

    info = Dict(
        :steps => Int[],
        :episodes => Int[],
        :returns => Float64[],
        :returns_error => Float64[]
    )

    input_dims = (1,500)
    na = 3
    buff_cap = 100_000
    buffer = DataBuffer(input_dims, na, buff_cap)

    for itr in 1:n_iter+1
        @info itr <= n_iter ? "Gathering Data - Iteration: $itr" : "Gathering Data - Final Evaluation"
        returns, steps = gen_data(pomdp, net, params, buffer)
        
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

        # This is so we get the stats after the n_iter network update
        itr == n_iter+1 && break

        @info "Training"

        train_data = (; 
            x = select_last_dim(buffer.network_input, 1:buffer.length),
            value_target = select_last_dim(buffer.value_target, 1:buffer.length),
            policy_target = select_last_dim(buffer.policy_target, 1:buffer.length),
        )

        net, train_info = train!(net, train_data, params)

        if plot_training
            pv = plot(train_info[:train_R]; c=1, label="train", ylabel="FVU", title="Training Information")
            plot!(pv, train_info[:valid_R]; c=2, label="valid")
            pp = plot(train_info[:train_KL]; c=1, label="train", ylabel="Policy KL")
            plot!(pp, train_info[:valid_KL]; c=2, label="valid", xlabel="Epochs")
            plot(pv, pp; layout=(2,1)) |> display
        end
    end

    info[:steps] = cumsum(info[:steps])
    info[:episodes] = cumsum(info[:episodes])

    return net, info
end


end # module minBetaZero
