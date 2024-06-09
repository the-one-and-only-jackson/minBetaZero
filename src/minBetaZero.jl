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
    t_max       = 100
    n_episodes  = 500
    n_iter      = 20
    batchsize   = 128
    lr          = 3e-4
    value_scale = 1.0
    lambda      = 0.0
    plot_training = false
    train_device  = cpu
    input_dims = (1,)
    na = 3
    buff_cap = 10_000
    n_particles = 500
    n_planning_particles = 100
    train_on_planning_b = true
    use_belief_reward = true
    use_gumbel_target = true
    train_intensity = 8
    warmup_steps = buff_cap ÷ train_intensity
    GumbelSolver_args = (;
        tree_queries        = 100,
        n_particles         = n_planning_particles,
        k_o                 = 20.,
        check_repeat_obs    = false,
        resample            = true,
        cscale              = 1.,
        cvisit              = 50.,
        m_acts_init         = na
    )
end

include("test_policy_net.jl")
include("collect_threaded.jl")
include("collect_distributed.jl")

function betazero(params::minBetaZeroParameters, pomdp::POMDP, net)
    (; n_iter, input_dims, na, buff_cap, n_particles, n_planning_particles, train_on_planning_b, train_intensity, warmup_steps) = params

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

    buffer_particles = train_on_planning_b ? n_planning_particles : n_particles
    buffer = DataBuffer((input_dims..., buffer_particles), na, buff_cap)

    for itr in 1:n_iter+1
        iter_timer = time()

        @info itr <= n_iter ? "Gathering Data - Iteration: $itr" : "Gathering Data - Final Evaluation"

        if nprocs() > 1 # prioritize distributed ?
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

        # net_results = test_network(net, pomdp, params; n_episodes=100)
        # returns_mean = mean(net_results)
        # returns_error = std(net_results)/sqrt(length(net_results))
        # push!.(
        #     (info[:network_returns], info[:network_returns_error]),
        #     (returns_mean, returns_error)
        # )
        # _rm = round(returns_mean; sigdigits=1+Base.hidigit(returns_mean,10)-Base.hidigit(returns_error,10))
        # _re = round(returns_error; sigdigits=1)
        # @info "Network mean return $_rm +/- $_re"

        # This is so wez get the stats after the n_iter network update
        if itr == 1 + n_iter
            println()
            break
        end

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

function collect_data_returns(storage_channel, n_episodes::Int)
    progress = Progress(n_episodes)
    ret_vec = zeros(Float32, n_episodes)
    steps = 0

    for i in 1:n_episodes
        data = take!(storage_channel)
        steps += length(data.value_target)
        ret_vec[i] = data.returns
        next!(progress)
    end
    
    close(storage_channel)

    return ret_vec, steps
end

function work_fun(pomdp, planner, params)
    (; t_max, n_particles, n_planning_particles, train_on_planning_b, use_belief_reward, use_gumbel_target) = params

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
        if n_planning_particles == n_particles
            b_querry = b
        else
            b_perm = randperm(n_particles)[1:n_planning_particles]
            b_querry = ParticleCollection(particles(b)[b_perm])
        end
        
        a, a_info = action_info(planner, b_querry)
        aid = actionindex(pomdp, a)
        s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)

        b_target = train_on_planning_b ? b_querry : b

        push!.((b_vec, aid_vec, state_reward), (b_target, aid, r))

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

function train!(net, data, params::minBetaZeroParameters)
    (; batchsize, lr, lambda, train_device, plot_training, value_scale) = params

    value_scale = Float32(value_scale)

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

            return value_scale * losses.value_loss + losses.policy_loss
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

end