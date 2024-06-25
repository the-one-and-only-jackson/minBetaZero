using POMDPs, POMDPTools, ParticleFilters
using Statistics, StatsBase, Distributions, Random, SparseArrays
using Distributed
using Flux, CUDA
using BenchmarkTools

includet("lasertag.jl")
includet("exact_mdp.jl")
includet("AZTrees/AZTrees.jl")
using .LaserTag, .AZTrees

includet("ac_buffer.jl")
includet("history.jl")
includet("az_agent.jl")
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

    mcts_params = (; m_acts_init=2, tree_querries=4)
    worker = Worker(; mdp, actor_critic, n_agents=512, batchsize=128, mcts_params...);

    @time worker_main(worker; n_steps=10_000) # as low as 3.2 s, 5.75 G
    @time GC.gc() # as high as 725 s

    return
end

test_main() # 6 GB alloc, 3.5 s w/o GC

GC.gc(true)


querry_ready.(worker.batch_manager.batches)
response_ready.(worker.batch_manager.batches)
