module minBetaZero

using Flux, CUDA
using POMDPs, POMDPModelTools, POMDPTools
using ParticleFilters
using Statistics, Distributions, Random
using Pkg
using ProgressMeter, Plots

# include("../lib/ParticleFilterTrees/src/ParticleFilterTrees.jl")
using ParticleFilterTrees
export PFTDPWTree, PFTDPWSolver, SparsePFTSolver, PFTDPWPlanner, PFTBelief 

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
    PFTDPWSolver_args = (;
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
        cvisit              = 50.
    )
    t_max       = 100
    n_episodes  = 500
    n_workers   = 100
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

function gen_data(pomdp, net, params::minBetaZeroParameters)
    (; n_episodes, n_workers) = params

    progress = Progress(n_episodes)
    # channel = RemoteChannel(()->Channel{Int}(), 1)
    channel = Channel{Int}(1)

    total_steps = 0
    async_loop = Threads.@spawn while true
        step = take!(channel)
        if step == 0
            finish!(progress)
            break
        elseif total_steps < progress.n # i dont know how to deal with progressmeter correctly
            total_steps += step
            if total_steps <= progress.n
                next!(progress; step)
            else
                next!(progress; step=progress.n-(total_steps-step))
            end
        end
    end

    worker_querries = [Channel{Any}(1) for _ in 1:n_workers]
    master_responses = [Channel{Any}(1) for _ in 1:n_workers]

    function work_fun(i)
        worker_net = function(x; kwargs...)
            put!(worker_querries[i], x)
            take!(master_responses[i])
        end

        b_vec = []
        v_vec = Float32[]
        p_vec = []
        ret_vec = Float32[]

        episodes = 0
        while episodes < max(1,n_episodes/n_workers)
            episode_data = gen_episode(pomdp, worker_net, params)
            dst = (episode_data.b, episode_data.v, episode_data.p, episode_data.g)
            src = (b_vec, v_vec, p_vec, ret_vec)
            append!.(src, dst)

            episodes += 1
            put!(channel, 1)
        end

        return (; 
            value   = reshape(v_vec,1,:), 
            belief  = stack(input_representation, b_vec), 
            policy  = stack(p_vec),
            returns = ret_vec
        )
    end

    master_net = net |> gpu

    futures = [Threads.@spawn work_fun(i) for i in 1:n_workers]

    while !all(istaskdone, futures)
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

    data = fetch.(futures)

    put!(channel, 0)
    wait(async_loop)

    k = (:value, :belief, :policy, :returns)
    return NamedTuple{k}(reduce((args...)->cat(args...; dims=ndims(first(args))), (d[key] for d in data)) for key in k)
end

function gen_episode(pomdp, net, params::minBetaZeroParameters)
    (; t_max, PFTDPWSolver_args) = params

    planner = solve(
        PFTDPWSolver(;
            getpolicyvalue = function(x)
                out = net(input_representation(x))
                return (; value=out.value[], policy=vec(out.policy))
            end,
            PFTDPWSolver_args...
        ),
        pomdp
    )

    bmdp = ParticleBeliefMDP(pomdp)
    b = rand(initialstate(bmdp))

    b_vec = typeof(b)[]
    p_vec = Vector{Float32}[]
    r_vec = Float32[]
    p_nt = Float32[]
    term_flag = false
    for _ in 1:t_max
        a, a_info = action_info(planner, b)
        aid = actionindex(pomdp,a)
        p = Flux.onehot(aid, 1:length(actions(pomdp)))
        bp, r, info = @gen(:sp,:r,:info)(bmdp, b, aid)
        push!.((b_vec,p_vec,r_vec,p_nt), (b,p,r,info))
        b = bp
        if ParticleFilterTrees.isterminalbelief(b)
            term_flag = true
            break
        end
    end

    gamma = discount(bmdp)
    for i in length(r_vec)-1:-1:1
        r_vec[i] += p_nt[i] * gamma * r_vec[i+1]
    end

    episode_return = r_vec[1]

    return (; b=b_vec, v=r_vec, p=p_vec, g=episode_return)
end

function train!(net, data, params::minBetaZeroParameters)
    (; train_frac, batchsize, lr, lambda, n_epochs, train_dev, early_stop, n_warmup) = params

    split_data = Flux.splitobs(data, at=train_frac) 

    Etrain = mean(-sum(x->iszero(x) ? x : x*log(x), split_data[1].policy_target; dims=1))
    Evalid = mean(-sum(x->iszero(x) ? x : x*log(x), split_data[2].policy_target; dims=1))
    varVtrain = var(split_data[1].value_target)
    varVvalid = var(split_data[2].value_target)

    split_data = split_data |> train_dev
    train_data = Flux.DataLoader(split_data[1]; batchsize=min(batchsize, Flux.numobs(split_data[1])), shuffle=true, partial=false)
    valid_data = Flux.DataLoader(split_data[2]; batchsize=min(4*batchsize, Flux.numobs(split_data[2])), shuffle=true, partial=false)

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
            lr_scale = min(grad_steps / n_warmup, 1 - (grad_steps - n_warmup)/(n_epochs*length(train_data) - n_warmup))
            Flux.adjust!(opt; eta=lr_scale*lr, lambda=lr_scale*lr*lambda)    
    
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
        if epoch - last_checkpoint > early_stop
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

    @info "Number of processes: $(nworkers())"

    info = Dict(
        :steps => Int[],
        :episodes => Int[],
        :returns => Float64[],
        :returns_error => Float64[]
    )

    for itr in 1:n_iter+1
        @info itr <= n_iter ? "Gathering Data - Iteration: $itr" : "Gathering Data - Final Evaluation"
        data = gen_data(pomdp, net, params)
        
        steps = length(data.value)
        episodes = length(data.returns)
        returns_mean = mean(data.returns)
        returns_error = std(data.returns)/sqrt(length(data.returns))

        push!.((info[:steps], info[:episodes], info[:returns], info[:returns_error]), (steps, episodes, returns_mean, returns_error))

        _rm = round(returns_mean; sigdigits=1+Base.hidigit(returns_mean,10)-Base.hidigit(returns_error,10))
        _re = round(returns_error; sigdigits=1)
        @info "Mean return $_rm +/- $_re"
        @info "Gathered $steps data points over $episodes episodes"

        # This is so we get the stats after the n_iter network update
        itr == n_iter+1 && break

        @info "Training"
        net, train_info = train!(net, (; x=data.belief, value_target=data.value, policy_target=data.policy), params)

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
