using POMDPs, POMDPTools, ParticleFilters
using Statistics, StatsBase, Distributions, Random, SparseArrays
using Flux, CUDA
using Plots

includet("cartpole.jl")
using .CartPole

includet("AlphaZero/AlphaZero.jl")
using .AlphaZero

includet("plot_smoothing.jl")

mdp = CartPoleMDP()
na = length(actions(mdp))
input_dims = size(rand(MersenneTwister(1), initialstate(mdp)))
ns = prod(input_dims)

params = AlphaZeroParams(;
    buff_cap = 100_000,
    warmup_steps = 10_000,
    steps_per_iter = 10_000,
    inference_batchsize = 128,
    batchsize = 1024,
    lr = 1e-3,
    lambda = 1e-2,
    n_iter = 100,
    inference_T = Float32,
    value_scale = 0.5,
    plot_training = false,
    tree_queries = 2,
    k_o = 2,
    max_steps = 500,
    segment_length = typemax(Int)
)

nn_params = NetworkParameters(
    action_size         = na,
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(range(0, 100, length=256)),
    p_dropout           = 0.1,
    neurons             = 256,
    hidden_layers       = 1,
    shared_net          = Chain(
                            Flux.Scale(ns),
                            Flux.Dense(ns, 256, tanh),
                            Flux.Dense(256, 256, tanh)
                        ),
    shared_out_size     = (256,),
    activation          = tanh
)

ac = ActorCritic(nn_params)
ac.shared[1].scale .= [10, 2, 10, 2]

net, info = alphazero(params, mdp, ac);

plot_smoothing(info[:steps], info[:episode_length]; k=10)

info[:training][1]
