using minBetaZero

using POMDPs
using POMDPTools
using ParticleFilters
using Flux
using Statistics
using Plots

include("models/LightDark.jl")
using .LightDark

function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
    rep = Float32[p.y for p in particles(b)]
    reshape(rep, 1, :)
end

mean_std_layer(x) = dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)

params = minBetaZeroParameters(
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
    ),
    t_max = 100,
    n_episodes = 256,
    n_workers = 256,
    n_iter = 10,
    train_frac = 0.8,
    batchsize = 128,
    lr = 10e-4,
    lambda = 0.0,
    n_epochs = 50,
    plot_training = true,
    train_dev = gpu,
    early_stop = 10,
    n_warmup = 500
)

nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size=3,
    input_size=(1,),
    critic_loss = Flux.Losses.logitcrossentropy,
    critic_categories = collect(-100:10:100),
    p_dropout = 0.1,
    neurons = 64,
    hidden_layers = 2,
    shared_net = mean_std_layer,
    shared_out_size = (2,) # must manually set... fix at a later date...        
)


@time net, info = betazero(params, LightDarkPOMDP(), ActorCritic(nn_params));

plot(
    plot(info[:steps], info[:returns]; label=false, xlabel="Steps", title="Mean Episodic Return"),
    plot(info[:episodes], info[:returns]; label=false, xlabel="Episodes");
    layout=(2,1)
)



net




