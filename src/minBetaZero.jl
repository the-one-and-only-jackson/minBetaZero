module minBetaZero

using Distributed

using Flux
using POMDPs, POMDPModelTools, POMDPTools
using ParticleFilters, MCTS
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

include("pft_interface.jl")
export NetworkWrapper, PUCT, get_value, get_policy, input_representation

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
        k_a                 = 5.,
        alpha_a             = 0.,
        enable_action_pw    = false,
        k_o                 = 24.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 1_000, 
        beliefcache_size    = 1_000,
    )
    t_max = 100
    n_episodes = 500
    n_steps = 5000
    use_episodes = true # gather data for at least n_episodes or at least n_steps
    c_puct = 100.0
    n_iter = 20
    noise_alpha = 0.25
    noise_param = 0.1
    train_frac = 0.8
    batchsize = 128
    lr = 3e-4
    lambda = 0.0
    n_epochs = 50
    plot_training = false
end

function gen_data(pomdp, net, params::minBetaZeroParameters)
    (; use_episodes, n_steps, n_episodes) = params

    nw = NetworkWrapper(; net)

    planner = solve(
        PFTDPWSolver(;
            value_estimator  = nw,
            policy_estimator = PUCT(; net=nw, c=params.c_puct),
            params.PFTDPWSolver_args...
        ),
        pomdp
    )

    progress = Progress(use_episodes ? n_episodes : n_steps)
    channel = RemoteChannel(()->Channel{Int}(), 1)

    total_steps = 0
    async_loop = @async while true
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

    data = pmap(1:nworkers()) do i
        b_vec = []
        v_vec = Float32[]
        p_vec = Vector{Float32}[]
        ret_vec = Float32[]

        episodes = 0
        steps = 0

        while (use_episodes && episodes < n_episodes/nworkers()) || (!use_episodes && steps < n_steps/nworkers())
            episode_data = gen_episode(pomdp, nw, planner, params)
            dst = (episode_data.b, episode_data.v, episode_data.p, episode_data.g)
            src = (b_vec, v_vec, p_vec, ret_vec)
            append!.(src, dst)

            episodes += 1
            steps += length(episode_data.v)

            if use_episodes
                put!(channel, 1)
            else
                put!(channel, length(episode_data.v))
            end
        end

        return (; 
            value   = reshape(v_vec,1,:), 
            belief  = stack(input_representation, b_vec), 
            policy  = stack(p_vec),
            returns = ret_vec
        )
    end

    put!(channel, 0)
    wait(async_loop)

    k = (:value, :belief, :policy, :returns)
    return NamedTuple{k}(reduce((args...)->cat(args...; dims=ndims(first(args))), (d[key] for d in data)) for key in k)
end

function gen_episode(pomdp, nw, planner, params::minBetaZeroParameters)
    t_max = params.t_max

    bmdp = ParticleBeliefMDP(pomdp)
    b = rand(initialstate(bmdp))

    b_vec = typeof(b)[]
    p_vec = Vector{Float32}[]
    r_vec = Float32[]
    term_flag = false
    for _ in 1:t_max
        empty!(nw)
        _, a_info = action_info(planner, b)
        p = calculate_targetdist(pomdp, a_info.tree)
        a_idx = sample_cat(p) # bmdp uses action indexes
        bp, r = @gen(:sp,:r)(bmdp, b, a_idx)
        push!.((b_vec,p_vec,r_vec), (b,p,r))
        b = bp
        if ParticleFilterTrees.isterminalbelief(b)
            term_flag = true
            break
        end
    end

    r_last = bootstrap_val = term_flag ? zero(eltype(r_vec)) : get_value(nw, b)
    gamma = discount(bmdp)
    for i in length(r_vec):-1:1
        r_last = r_vec[i] += gamma * r_last
    end

    episode_return = r_vec[1] - bootstrap_val * gamma^t_max

    return (; b=b_vec, v=r_vec, p=p_vec, g=episode_return)
end

function calculate_targetdist(pomdp, tree; zq=1, zn=1)
    # P = Vector{Float32}(undef, length(actions(pomdp)))
    P = zeros(Float32, length(actions(pomdp)))
    qmax = -Inf
    nsum = 0.0
    for (_,aid) in tree.b_children[1]
        qmax = max(qmax, tree.Qha[aid])
        nsum += tree.Nha[aid]
    end
    for (a,aid) in tree.b_children[1]
        n_norm = tree.Nha[aid] / nsum
        q_norm = exp(tree.Qha[aid] - qmax)
        P[actionindex(pomdp,a)] = n_norm^zn * q_norm^zq
    end
    @assert all(isfinite, P) "$P, $(tree.Qha[1]), $(tree.Nha[1])"
    P ./= sum(P)
    @assert all(0 .<= P .<= 1) "$P, $(tree.Qha[1]), $(tree.Nha[1])"
    P
end

function train!(net, data, params::minBetaZeroParameters)
    (; train_frac, batchsize, lr, lambda, n_epochs) = params

    split_data = Flux.splitobs(data, at=train_frac)
    train_data = Flux.DataLoader(split_data[1]; batchsize=min(batchsize, Flux.numobs(split_data[1])), shuffle=true, partial=false)
    valid_data = Flux.DataLoader(split_data[2]; batchsize=min(4*batchsize, Flux.numobs(split_data[2])), shuffle=true, partial=false)

    opt = Flux.setup(Flux.Optimiser(Flux.Adam(lr), WeightDecay(lr*lambda)), net)

    Etrain = mean(-sum(x->iszero(x) ? x : x*log(x), train_data.data.policy_target; dims=1))
    Evalid = mean(-sum(x->iszero(x) ? x : x*log(x), valid_data.data.policy_target; dims=1))
    varVtrain = var(train_data.data.value_target)
    varVvalid = var(valid_data.data.value_target)

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
    
    @showprogress for _ in 1:n_epochs
        Flux.trainmode!(net)
        for (; x, value_target, policy_target) in train_data    
            grads = Flux.gradient(net) do net
                lossfun(net, x; value_target, policy_target, info=train_info, policy_entropy=Etrain, value_var=varVtrain)
            end
            Flux.update!(opt, net, grads[1])
        end

        Flux.testmode!(net)
        for (; x, value_target, policy_target) in valid_data    
            lossfun(net, x; value_target, policy_target, info=valid_info, policy_entropy=Evalid, value_var=varVvalid)
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
        info[k] = dropdims(mean(reshape(v,:,n_epochs); dims=1); dims=1)
    end

    return net, info
end

function betazero(params::minBetaZeroParameters, pomdp::POMDP, net)
    (; n_iter, noise_alpha, noise_param, plot_training) = params

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

        noise = rand(Dirichlet(size(data.policy,1), Float32(noise_param)), size(data.policy,2))
        alpha = Float32(noise_alpha)
        noisy_policy = (1-alpha)*data.policy + alpha*noise

        training_data = (; x=data.belief, value_target=data.value, policy_target=noisy_policy)

        @info "Training"
        net, train_info = train!(net, training_data, params)

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
