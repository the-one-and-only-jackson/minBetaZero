module ParticleFilterTrees

using POMDPs
import POMDPTools
using Random
using MCTS
using PushVectors
using ParticleFilters

export PFTDPWTree, PFTDPWSolver, SparsePFTSolver, PFTDPWPlanner, PFTBelief
# export FastRandomSolver, FastRandomRolloutEstimator

include("cache.jl")
include("pushvector.jl")
include("pftbelief.jl")
include("tree.jl")


"""
...
- `max_depth::Int = 20` - Maximum tree search depth
- `n_particles::Int = 100` - Number of particles representing belief
- `k_o::Float64 = 10.0` - Initial observation widening parameter
- `alpha_o::Float64 = 0.0` - Observation progressive widening parameter
- `k_a::Float64 = 5.0` - Initial action widening parameter
- `alpha_a::Float64 = 0.0` - Action progressive widening parameter
- `criterion = MaxPoly()` - action selection criterion
- `tree_queries::Int = 1_000` - Maximum number of tree search iterations
- `max_time::Float64 = Inf` - Maximum tree search time (in seconds)
- `rng = Random.default_rng()` - Random number generator
- `value_estimator = FastRandomSolver()` - Belief node value estimator
- `check_repeat_obs::Bool = true` - Check that repeat observations do not overwrite beliefs (added dictionary overhead)
- `enable_action_pw::Bool = false` - Alias for `alpha_a = 0.0`
- `beliefcache_size::Int = 1_000` - Number of particle/weight vectors to cache offline
- `treecache_size::Int = 1_000` - Number of belief/action nodes to preallocate in tree (reduces `Base._growend!` calls)
- `default_action = RandomDefaultAction()` - Action to take if root has no children
...
"""
Base.@kwdef struct PFTDPWSolver{CRIT, RNG<:AbstractRNG, DA} <: Solver
    tree_queries::Int       = 1_000
    max_time::Float64       = Inf # (seconds)
    max_depth::Int          = 20
    n_particles::Int        = 100
    k_o::Float64            = 10.0
    alpha_o::Float64        = 0.0 # Observation Progressive widening parameter
    k_a::Float64            = 5.0
    alpha_a::Float64        = 0.0 # Action Progressive widening parameter
    criterion::CRIT         = MaxPoly(c=1.0) # used in policy_estimator
    rng::RNG                = Random.default_rng()
    value_estimator::Any    = FastRandomSolver()
    check_repeat_obs::Bool  = true
    enable_action_pw::Bool  = false
    beliefcache_size::Int   = 1_000
    treecache_size::Int     = 1_000
    default_action::DA      = RandomDefaultAction()
    policy_estimator::Any   = BasicActionSelector
    resample::Bool          = true
end

struct PFTDPWPlanner{SOL<:PFTDPWSolver, M<:POMDP, TREE<:PFTDPWTree, VE, S, PE} <: Policy
    pomdp::M
    sol::SOL
    tree::TREE
    solved_VE::VE
    cache::BeliefCache{S}
    solved_PE::PE
end

SparsePFTSolver(;kwargs...) = PFTDPWSolver(;kwargs..., alpha_o=0.0, alpha_a=0.0, enable_action_pw=false)

function POMDPs.solve(sol::PFTDPWSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    solved_ve = MCTS.convert_estimator(sol.value_estimator, sol, pomdp)
    solved_pe = sol.policy_estimator(sol, pomdp)

    @assert sol.enable_action_pw || length(actions(pomdp)) < Inf "Action space should have some defined length if enable_action_pw=false"

    cache = BeliefCache{S}(min(sol.tree_queries, sol.beliefcache_size), sol.n_particles)

    tree = PFTDPWTree{S,A,O}(
        min(sol.tree_queries, sol.treecache_size), 
        sol.check_repeat_obs, 
        sol.k_o, 
        sol.k_a
        )

    return PFTDPWPlanner(pomdp, sol, tree, solved_ve, cache, solved_pe)
end

POMDPs.action(planner::PFTDPWPlanner, b) = first(action_info(planner, b))
function POMDPTools.action_info(planner::PFTDPWPlanner, b)
    t0 = time()
    tree = planner.tree

    free!(planner.cache)

    s,w = gen_empty_belief(planner.cache, planner.sol.n_particles)
    particle_b = initialize_belief!(planner.sol.rng, s, w, planner.pomdp, b)
    insert_root!(tree, particle_b)
    
    iter = 0
    while (time()-t0 < planner.sol.max_time) && (iter < planner.sol.tree_queries)
        mcts_main(planner)
        iter += 1
    end

    a = if isempty(first(tree.b_children))
        planner.sol.default_action(planner.pomdp, b)
    else
        first(select_action(MaxQ(), tree, 1, 0))
    end

    info = (
        n_iter = iter,
        tree   = tree,
        time   = time() - t0,
        Q_root = Dict(a=>tree.Qha[aid] for (a,aid) in tree.b_children[1]),
        N_root = Dict(a=>tree.Nha[aid] for (a,aid) in tree.b_children[1])
    )

    return a, info
end

function mcts_main(planner::PFTDPWPlanner, b_idx::Int=1, d::Int=0) # d is current depth, not max depth
    (; tree, pomdp, sol, cache, solved_VE, solved_PE) = planner
    (; max_depth, k_o, alpha_o, rng, check_repeat_obs, resample) = sol

    if d==max_depth || isterminalbelief(tree.b[b_idx])
        return 0.0
    end

    # select action
    a, ba_idx = select_action(solved_PE, tree, b_idx, d)

    # observation/belief widening
    if length(tree.ba_children[ba_idx]) ≤ k_o*tree.Nha[ba_idx]^alpha_o
        b = tree.b[b_idx]
        p_idx = non_terminal_sample(rng, pomdp, b)
        sample_s = particle(b, p_idx)
        sample_sp, o, sample_r = @gen(:sp,:o,:r)(pomdp, sample_s, a, rng)
    
        if check_repeat_obs && haskey(tree.bao_children, (ba_idx, o))
            bp_idx = tree.bao_children[(ba_idx,o)]
            push!(tree.ba_children[ba_idx], bp_idx)
            r_togo = mcts_main(planner, bp_idx, d+1)
        else
            bp_particles, bp_weights = gen_empty_belief(cache, n_particles(b))
            bp, r, nt_prob = GenBelief(
                rng, bp_particles, bp_weights, cache.resample, 
                pomdp, b, a, o, p_idx, sample_sp, sample_r, resample
            )
            bp_idx = insert_belief!(tree, bp, ba_idx, o, r, nt_prob, check_repeat_obs)
            r_togo = isterminalbelief(bp) ? 0.0 : MCTS.estimate_value(solved_VE, pomdp, bp, max_depth-(d+1))
        end
    else
        bp_idx = rand(rng, tree.ba_children[ba_idx])
        r_togo = mcts_main(planner, bp_idx, d+1)
    end

    total = tree.b_rewards[bp_idx] + discount(pomdp) * tree.b_ntprob[bp_idx] * r_togo

    # update tree
    tree.Nh[b_idx] += 1
    tree.Nha[ba_idx] += 1
    tree.Qha[ba_idx] += (total - tree.Qha[ba_idx]) / tree.Nha[ba_idx]

    # return sum of rewards
    return total::Float64
end

include("domain_knowledge.jl")
include("rollout.jl")

end 