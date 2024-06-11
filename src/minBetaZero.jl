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

function setup()
    Pkg.develop(PackageSpec(url=joinpath(@__DIR__,"..","lib","ParticleFilterTrees")))
end

@kwdef struct minBetaZeroParameters
    t_max               = 100
    n_episodes          = 500
    n_iter              = 20
    inference_batchsize = 32
    input_dims          = (1,)
    na                  = 3
    n_particles         = 500
    n_planning_particles= 100
    train_on_planning_b = true
    use_belief_reward   = true
    use_gumbel_target   = true
    num_itr_stored      = 1

    GumbelSolver_args   = (;
        tree_queries        = 100,
        n_particles         = n_planning_particles,
        k_o                 = 20.,
        check_repeat_obs    = false,
        resample            = true,
        cscale              = 1.,
        cvisit              = 50.,
        m_acts_init         = na
    )

    batchsize       = 128
    lr              = 3e-4
    value_scale     = 1.0
    lambda          = 0.0
    plot_training   = false
    train_device    = cpu
    buff_cap        = t_max * n_episodes * num_itr_stored
    train_intensity = 8
    warmup_steps    = 0
end

include("DataBuffer.jl")
include("test_policy_net.jl")
include("collect_shared.jl")
include("collect_threaded.jl")
include("collect_distributed.jl")

function betazero(params::minBetaZeroParameters, pomdp::POMDP, net)
    (; n_iter, input_dims, na, buff_cap, n_particles, n_planning_particles, 
    train_on_planning_b, train_intensity, warmup_steps, num_itr_stored) = params

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

    buffer_particles = train_on_planning_b ? n_planning_particles : n_particles
    buffer = DataBuffer((input_dims..., buffer_particles), na, buff_cap)

    _steps_saved = 0

    for itr in 1:n_iter+1
        iter_timer = time()

        (mod(itr-1, num_itr_stored) == 0) && reset_buffer!(buffer)

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

        # This is so we get the stats after the n_iter network update
        if itr == 1 + n_iter
            println()
            break
        end

        if buffer.length >= warmup_steps
            _steps_saved += steps * train_intensity
            n_batches = _steps_saved ÷ params.batchsize
            _steps_saved -= n_batches * params.batchsize

            if n_batches > 0
                net, train_info = train!(net, buffer, params, n_batches)
                push!(info[:training], train_info)
            end
        end

        iter_time = ceil(Int, time()-iter_timer)
        @info "Iteration time: $iter_time seconds"
        println()
    end

    info[:steps] = cumsum(info[:steps])
    info[:episodes] = cumsum(info[:episodes])

    return net, info
end

function train!(net, buffer::DataBuffer, params::minBetaZeroParameters, n_batches::Int)
    (; batchsize, lr, lambda, train_device, plot_training, value_scale) = params

    value_scale = Float32(value_scale)

    reset_minibatch(buffer)

    net = train_device(net)
    opt = Flux.setup(Flux.Optimiser(Flux.Adam(lr), WeightDecay(lr*lambda)), net)

    info = Dict(
        :policy_loss => Float32[], 
        :policy_KL   => Float32[], 
        :value_loss  => Float32[], 
        :value_FVU   => Float32[]
    )
    
    Flux.trainmode!(net) 
    for _ in 1:n_batches
        data = sample_minibatch(buffer, batchsize) |> train_device

        Etrain = mean(-sum(x->iszero(x) ? x : x*log(x), data.policy_target; dims=1))
        varVtrain = var(data.value_target)

        grads = Flux.gradient(net) do net
            losses = getloss(net, data.network_input; data.value_target, data.policy_target)

            Flux.Zygote.ignore_derivatives() do 
                push!(info[:policy_loss], losses.policy_loss          )
                push!(info[:policy_KL]  , losses.policy_loss - Etrain )
                push!(info[:value_loss] , losses.value_loss           )
                push!(info[:value_FVU]  , losses.value_mse / varVtrain)
                return nothing
            end

            total_loss = value_scale * losses.value_loss + losses.policy_loss

            return total_loss
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