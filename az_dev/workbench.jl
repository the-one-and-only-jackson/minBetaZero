using POMDPs, POMDPTools, ParticleFilters
using Statistics, StatsBase, Distributions, Random, SparseArrays
using Flux, CUDA
using BenchmarkTools

includet("lasertag.jl")
includet("exact_mdp.jl")
includet("AZTrees/AZTrees.jl")
using .LaserTag, .AZTrees

includet("ac_buffer.jl")
includet("history.jl")
includet("az_worker.jl")

function test_main()
    mdp = ExactBeliefMDP(LaserTagPOMDP())
    na = length(actions(mdp))
    ns = length(rand(initialstate(mdp)))

    actor_critic = Chain(
        Dense(ns => 64, relu),
        Parallel(
            (value, policy) -> (; value, policy),
            Dense(64 => 1),
            Dense(64 => na)
        )
    )

    mcts_params = (; m_acts_init=2, tree_querries=10, k_o=5)
    worker = Worker(; mdp, actor_critic, n_agents=512, batchsize=128, mcts_params...);

    @time episode_returns, histories = worker_main(worker; n_steps=10_000) # as low as 3.2 s, 5.75 G

    return nothing
end

GC.enable_logging(true)

test_main(); # 6 GB alloc, 3.5 s w/o GC

GC.gc()


querry_ready.(worker.batch_manager.batches)
response_ready.(worker.batch_manager.batches)



mdp = ExactBeliefMDP(LaserTagPOMDP())
na = length(actions(mdp))
ns = length(rand(initialstate(mdp)))

actor_critic(x) = x

mcts_params = (; m_acts_init=2, tree_querries=4)
worker = Worker(; mdp, actor_critic, n_agents=32, batchsize=32, mcts_params...);

querry_ready(worker.batch_manager)

test(worker)

function test(worker)
    @time process_batch!(worker.batch_manager, worker.actor_critic)
end
