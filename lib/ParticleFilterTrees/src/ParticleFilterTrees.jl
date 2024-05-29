module ParticleFilterTrees

using POMDPs
import POMDPTools
using Random
using PushVectors
using ParticleFilters
using Distributions
using NNlib: softmax

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
Base.@kwdef struct PFTDPWSolver{RNG<:AbstractRNG, DA, F} <: Solver
    tree_queries::Int       = 1_000
    max_time::Float64       = Inf # (seconds)
    max_depth::Int          = 20
    n_particles::Int        = 100
    k_o::Float64            = 10.0
    alpha_o::Float64        = 0.0 # Observation Progressive widening parameter
    rng::RNG                = Random.default_rng()
    check_repeat_obs::Bool  = true
    beliefcache_size::Int   = 1_000
    treecache_size::Int     = 1_000
    default_action::DA      = (pomdp::POMDP, ::Any) -> rand(actions(pomdp))
    resample::Bool          = true
    cscale::Float64 = 1.
    cvisit::Float64 = 50.
    getpolicyvalue::F
end

struct PFTDPWPlanner{SOL<:PFTDPWSolver, M<:POMDP, TREE<:PFTDPWTree, S} <: Policy
    pomdp::M
    sol::SOL
    tree::TREE
    cache::BeliefCache{S}
end

function POMDPs.solve(sol::PFTDPWSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    cache = BeliefCache{S}(min(sol.tree_queries, sol.beliefcache_size), sol.n_particles)

    tree = PFTDPWTree{S,A,O}(
        min(sol.tree_queries, sol.treecache_size),
        length(actions(pomdp)),
        sol.check_repeat_obs, 
        sol.k_o
    )

    return PFTDPWPlanner(pomdp, sol, tree, cache)
end

POMDPs.action(planner::PFTDPWPlanner, b) = first(action_info(planner, b))
function POMDPTools.action_info(planner::PFTDPWPlanner, b)
    t0 = time()
    (; tree, cache, sol, pomdp) = planner
    (; rng, n_particles, rng, max_time, tree_queries, default_action, getpolicyvalue, cscale, cvisit) = sol

    free!(cache)

    s,w = gen_empty_belief(cache, n_particles)
    particle_b = initialize_belief!(rng, s, w, pomdp, b)

    insert_root!(tree, particle_b)
    P = getpolicyvalue(particle_b).policy
    for a in actions(pomdp)
        insert_action!(tree, 1, a, Float64(P[actionindex(pomdp,a)]))
    end

    live_actions = trues(length(actions(pomdp)))

    g_P = P + rand(rng, Gumbel(), length(actions(pomdp))) # g(a) + logits(a)

    m_acts_init = length(actions(pomdp)) # hyperparameter
    top_a_idx = sortperm(g_P; rev=true)
    for i in m_acts_init+1:length(actions(pomdp))
        live_actions[top_a_idx[i]] = false
    end
    m_acts_init
    ktarget = 1
    Ntarget = floor(Int, tree_queries/(ceil(log2(m_acts_init)) * nextpow(2,m_acts_init))) # under_itr

    iter = 0
    while (time()-t0 < max_time) && (iter < tree_queries) && count(live_actions) > 1
        sra = select_root_action(tree, pomdp, live_actions, Ntarget)
        if isnothing(sra)
            ktarget *= 2
            Ntarget += floor(Int, ktarget*tree_queries/(ceil(log2(m_acts_init)) * nextpow(2,m_acts_init)))
            reduce_root_actions(cscale, cvisit, Ntarget, live_actions, tree, pomdp, g_P)
            sra = select_root_action(tree, pomdp, live_actions, Ntarget)
        end
        a, ba_idx = sra

        mcts_main(planner, 1, a, ba_idx, 0, Ntarget)

        iter += 1
    end

    a = if isempty(first(tree.b_children))
        default_action(pomdp, b)
    else
        pomdp_aidx = findfirst(live_actions)
        actions(pomdp)[findfirst(a->actionindex(pomdp,a)==pomdp_aidx, actions(pomdp))]
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

function select_root_action(tree, pomdp, live_actions, Ntarget)
    for (a,ba_idx) in tree.b_children[1]
        if live_actions[actionindex(pomdp, a)] && tree.Nha[ba_idx] < Ntarget
            return (a, ba_idx)
        end
    end
    return nothing
end

function reduce_root_actions(cscale, cvisit, Ntarget, live_actions, tree, pomdp, g_P)
    sigma = cscale * (cvisit + Ntarget)
    for i in 1:floor(Int, count(live_actions)/2)
        local aidx_min::Int
        valmin = Inf
        for (a,ba_idx) in tree.b_children[1]
            pomdp_aidx = actionindex(pomdp, a)
            !live_actions[pomdp_aidx] && continue
            val = g_P[pomdp_aidx] + sigma * tree.Qha[ba_idx]
            if val < valmin
                valmin = val
                aidx_min = pomdp_aidx
            end
        end
        live_actions[aidx_min] = false
    end
    nothing
end

function mcts_main(planner::PFTDPWPlanner, b_idx::Int, d::Int, Ntarget::Int)
    (; tree, pomdp, sol) = planner
    (; cscale, cvisit) = sol

    sigma = cscale * (cvisit + Ntarget)
    v = tree.b_estval[b_idx]

    new_logits = zeros(length(actions(pomdp)))
    for (a, ba_idx) in tree.b_children[b_idx]
        pomdp_aidx = actionindex(pomdp,a)
        q_completed = tree.Nha[ba_idx] > 0 ? tree.Qha[ba_idx] : v
        sigma_q = sigma * q_completed
        logits = tree.Pha[ba_idx]
        new_logits[pomdp_aidx] = sigma_q + logits
    end
    pi_completed = softmax(new_logits)

    opt_val = -Inf
    opt_a = nothing
    opt_idx = 0
    Nh_1 = 1 + tree.Nh[b_idx]
    for (a,ba_idx) in tree.b_children[b_idx]
        pomdp_aidx = actionindex(pomdp,a)
        val = pi_completed[pomdp_aidx] - tree.Nha[ba_idx]/Nh_1
        if val > opt_val
            opt_val = val
            opt_a = a
            opt_idx = ba_idx
        end
    end

    mcts_main(planner, b_idx, opt_a, opt_idx, d, Ntarget)
end

function mcts_main(planner::PFTDPWPlanner, b_idx, a, ba_idx, d::Int, Ntarget::Int)
    (; tree, pomdp, sol) = planner
    (; max_depth, k_o, alpha_o, rng, check_repeat_obs) = sol

    if d==max_depth || isterminalbelief(tree.b[b_idx])
        return 0.0
    end

    # observation/belief widening
    if length(tree.ba_children[ba_idx]) ≤ k_o*tree.Nha[ba_idx]^alpha_o
        b = tree.b[b_idx]
        p_idx = non_terminal_sample(rng, pomdp, b)
        sample_s = particle(b, p_idx)
        sample_sp, o, sample_r = @gen(:sp,:o,:r)(pomdp, sample_s, a, rng)
    
        if check_repeat_obs && haskey(tree.bao_children, (ba_idx, o))
            bp_idx = tree.bao_children[(ba_idx,o)]
            push!(tree.ba_children[ba_idx], bp_idx)
            Vp = mcts_main(planner, bp_idx, d+1, Ntarget)
        else
            bp_idx, Vp = insert_new_belief!(planner, b, ba_idx, a, o, p_idx, sample_sp, sample_r)
        end
    else
        bp_idx = rand(rng, tree.ba_children[ba_idx])
        Vp = mcts_main(planner, bp_idx, d+1, Ntarget)
    end

    total = tree.b_rewards[bp_idx] + discount(pomdp) * tree.b_ntprob[bp_idx] * Vp

    # update tree
    tree.Nh[b_idx]   += 1
    tree.Nha[ba_idx] += 1
    tree.Qha[ba_idx] += (total - tree.Qha[ba_idx]) / tree.Nha[ba_idx]

    # return sum of rewards
    return total::Float64
end

function insert_new_belief!(planner, b, ba_idx, a, o, p_idx, sample_sp, sample_r)
    (; tree, pomdp, sol, cache) = planner
    (; max_depth, rng, check_repeat_obs, resample, getpolicyvalue) = sol

    bp_particles, bp_weights = gen_empty_belief(cache, n_particles(b))
    bp, r, nt_prob = GenBelief(
        rng, bp_particles, bp_weights, cache.resample, 
        pomdp, b, a, o, p_idx, sample_sp, sample_r, resample
    )

    if isterminalbelief(bp)
        bp_idx = insert_belief!(tree, bp, ba_idx, o, r, nt_prob, 0.0, check_repeat_obs)
        Vp = 0.0
    else
        (; value, policy) = getpolicyvalue(bp)
        Vp = Float64(value)

        bp_idx = insert_belief!(tree, bp, ba_idx, o, r, nt_prob, Vp, check_repeat_obs)

        for a in actions(pomdp)
            logit = Float64(policy[actionindex(pomdp,a)])
            insert_action!(tree, bp_idx, a, logit)
        end
    end

    return bp_idx, Vp
end


end 