using minBetaZero
using ParticleFilterTrees

using POMDPs
using POMDPTools
using ParticleFilters
using Flux
using Statistics
using Plots
using ProgressMeter

include("models/LightDark.jl")
using .LightDark

function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
    rep = Float32[p.y for p in particles(b)]
    reshape(rep, 1, :)
end


params = minBetaZeroParameters(
    GumbelSolver_args = (;
        max_depth           = 10,
        n_particles         = 100,
        tree_queries        = 100,
        max_time            = Inf,
        k_o                 = 20.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 1_000, 
        beliefcache_size    = 1_000,
        m_acts_init         = 3,
        cscale              = 10.0,
        cvisit              = 50.
    ),
    t_max           = 50,
    n_episodes      = 100,
    n_iter          = 20,
    batchsize       = 256,
    lr              = 3e-4,
    lambda          = 0.0,
    plot_training   = false,
    train_device    = gpu,
    inference_device = gpu,
    buff_cap = 50*100*4 # tmax * n_episodes * min_train_intensity
)

nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = 3,
    input_size          = (1,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(-100:10:100),
    p_dropout           = 0.1,
    neurons             = 64,
    hidden_layers       = 2,
    shared_net          = mean_std_layer,
    shared_out_size     = (2,) # must manually set... fix at a later date...        
)

struct netPolicyStoch{N,A} <: Policy
    net::N
    ordered_actions::A
end
POMDPs.action(p::netPolicyStoch, b) = p.ordered_actions[argmax(vec(p.net(input_representation(b)).policy) + rand(Gumbel(),3))]


pomdp = LightDarkPOMDP()

net = ActorCritic(nn_params)



(; n_iter, input_dims, na, buff_cap, n_particles) = params

buffer = minBetaZero.DataBuffer((input_dims..., n_particles), na, buff_cap)

returns, steps = minBetaZero.gen_data_distributed(pomdp, net, params, buffer; rand_policy = false)

episodes = length(returns)
returns_mean = mean(returns)
returns_error = std(returns)/sqrt(length(returns))

net = cpu(net)
net_results = minBetaZero.test_network(net, pomdp, params)
returns_mean = mean(net_results)
returns_error = std(net_results)/sqrt(length(net_results))

train_data = (; 
    x = minBetaZero.select_last_dim(buffer.network_input, 1:buffer.length),
    value_target = minBetaZero.select_last_dim(buffer.value_target, 1:buffer.length),
    policy_target = minBetaZero.select_last_dim(buffer.policy_target, 1:buffer.length),
)


net, train_info = train!(net, train_data, params)



(; batchsize, lr, lambda, train_device, plot_training) = params

data = train_data
Etrain = mean(-sum(x->iszero(x) ? x : x*log(x), data.policy_target; dims=1))
varVtrain = var(data.value_target)

train_batchsize = min(batchsize, Flux.numobs(data))
train_data = Flux.DataLoader(data; batchsize=train_batchsize, shuffle=true, partial=false) |> train_device


net = train_device(net)
opt = Flux.setup(Flux.Optimiser(Flux.Adam(lr), WeightDecay(lr*lambda)), net)

info = Dict(:policy_loss=>Float32[], :policy_KL=>Float32[], :value_loss=>Float32[], :value_FVU=>Float32[])
    

@showprogress for (; x, value_target, policy_target) in train_data
    Flux.trainmode!(net)
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

    Flux.testmode!(net)
    net_results = minBetaZero.test_network(cpu(net), pomdp, params)
    returns_mean = mean(net_results)
    returns_error = std(net_results)/sqrt(length(net_results))
    println(returns_mean, " ", returns_error)
end


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





