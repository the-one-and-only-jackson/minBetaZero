"""
...
- `n_particles::Int = 100` - Number of particles representing belief
- `k_o::Float64 = 10.0` - Initial observation widening parameter
- `alpha_o::Float64 = 0.0` - Observation progressive widening parameter
- `tree_queries::Int = 1_000` - Maximum number of tree search iterations
- `max_time::Float64 = Inf` - Maximum tree search time (in seconds)
- `rng = Random.default_rng()` - Random number generator
- `check_repeat_obs::Bool = true` - Check that repeat observations do not overwrite beliefs (added dictionary overhead)
- `beliefcache_size::Int = 1_000` - Number of particle/weight vectors to cache offline
- `treecache_size::Int = 1_000` - Number of belief/action nodes to preallocate in tree (reduces `Base._growend!` calls)
- `default_action = RandomDefaultAction()` - Action to take if root has no children
...
"""
Base.@kwdef struct GumbelSolver{RNG<:AbstractRNG, DA, F} <: Solver
    tree_queries::Int       = 1_000
    max_time::Float64       = Inf # (seconds)
    n_particles::Int        = 100
    k_o::Float64            = 10.0
    alpha_o::Float64        = 0.0 # Observation Progressive widening parameter
    rng::RNG                = Random.default_rng()
    check_repeat_obs::Bool  = false
    beliefcache_size::Int   = 1 + tree_queries
    treecache_size::Int     = 1 + tree_queries
    default_action::DA      = (pomdp::POMDP, ::Any) -> rand(rng, actions(pomdp))
    resample::Bool          = true
    cscale::Float64         = 1.
    cvisit::Float64         = 50.
    m_acts_init::Int
    getpolicyvalue::F
end

@kwdef struct GumbelPlanner{SOL<:GumbelSolver, M<:POMDP, TREE<:GuidedTree, S, OA, TN} <: Policy
    pomdp::M
    sol::SOL
    tree::TREE
    cache::BeliefCache{S}
    ordered_actions::OA = POMDPTools.ordered_actions(pomdp)
    live_actions::BitVector = falses(length(actions(pomdp)))
    q_extrema::Vector{Float64} = [Inf, -Inf]
    noisy_logits::Vector{Float64} = zeros(length(actions(pomdp)))
    target_N::TN = SeqHalf(; n=sol.tree_queries, m=sol.m_acts_init)
end

function get_dq(planner::GumbelPlanner; eps=1e-6)
    dq = planner.q_extrema[2] - planner.q_extrema[1]
    @assert isfinite(dq)
    return dq < eps ? one(dq) : dq
end

function update_dq!(planner::GumbelPlanner, q)
    if q < planner.q_extrema[1]
        planner.q_extrema[1] = q
    elseif q > planner.q_extrema[2]
        planner.q_extrema[2] = q
    end
    return nothing
end

@kwdef mutable struct SeqHalf
    const n::Int # sim budget
    const m::Int # initial number of actions
    N::Int = 0  # target number of sims per action
    k::Int = 0 # halving parameter
end

function next!(sh::SeqHalf)
    dN = (2 ^ sh.k) * sh.n / (sh.m * ceil(log2(sh.m)))
    sh.k += 1
    sh.N += max(1, floor(Int, dN))
    return sh.N
end

function reset!(sh::SeqHalf)
    sh.k = 0
    sh.N = 0
    next!(sh)
end


function POMDPs.solve(sol::GumbelSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    cache = BeliefCache{S}(min(sol.tree_queries+1, sol.beliefcache_size), sol.n_particles)

    tree = GuidedTree{S,Int,O}(
        min(sol.tree_queries+1, sol.treecache_size),
        length(actions(pomdp)),
        sol.check_repeat_obs, 
        sol.k_o
    )

    return GumbelPlanner(; pomdp, sol, tree, cache)
end

POMDPs.action(planner::GumbelPlanner, b) = first(POMDPTools.action_info(planner, b))

function POMDPTools.action_info(planner::GumbelPlanner, b)
    t0 = time()

    (; tree, cache, sol, pomdp, ordered_actions) = planner
    (; rng, n_particles, default_action) = sol

    free!(cache)

    s, w = gen_empty_belief(cache, n_particles)
    particle_b = initialize_belief!(rng, s, w, pomdp, b)

    if iszero(particle_b.non_terminal_ws)
        a = default_action(pomdp, b)
        info = (; n_iter = 0, tree = nothing, time = time() - t0, Q_root = nothing, N_root = nothing, policy_target = nothing)
        return a, info
    end

    n_iter = GumbelMCTS_main(planner, t0, particle_b)

    if isempty(first(tree.b_children))
        @warn "Taking random action"
        a = default_action(pomdp, b)
    else
        a = select_best_action(planner)
    end

    Q_root = Dict(ordered_actions[ai]=>tree.Qha[ba_idx] for (ai, ba_idx) in tree.b_children[1])
    N_root = Dict(ordered_actions[ai]=>tree.Nha[ba_idx] for (ai, ba_idx) in tree.b_children[1])

    policy_target = improved_policy(planner, 1)

    info = (; n_iter, tree, time = time() - t0, Q_root, N_root, policy_target)

    return a, info
end

function select_best_action(planner::GumbelPlanner)
    (; tree, sol, live_actions, noisy_logits, ordered_actions) = planner
    (; cscale, cvisit) = sol
    
    if count(live_actions) == 1
        ai_opt = findfirst(live_actions)
    else
        dq = get_dq(planner)
        Nmax = iszero(tree.Nh[1]) ? 0 : maximum(tree.Nha[ba_idx] for (_, ba_idx) in tree.b_children[1])
        sigma = cscale * (cvisit + Nmax) / dq
    
        ai_opt = 0
        valmax = -Inf
        for (ai, ba_idx) in tree.b_children[1]
            !live_actions[ai] && continue
            val = noisy_logits[ai] + sigma * tree.Qha[ba_idx]
            if val > valmax
                valmax = val
                ai_opt = ai
            end
        end
    end 

    return ordered_actions[ai_opt]
end

function GumbelMCTS_main(planner::GumbelPlanner, t0, particle_b)
    (; sol, target_N, ordered_actions, tree) = planner
    (; max_time, tree_queries) = sol

    initialize_root!(planner, particle_b)
    reset!(target_N)

    t_max = t0 + max_time

    for iter in 1:tree_queries
        (time() >= t_max) && return iter - 1

        ai, ba_idx = select_root_action(planner)
        a = ordered_actions[ai]

        mcts_main(planner, 1, a, ba_idx)
    end

    return tree_queries
end

function mcts_main(planner::GumbelPlanner, b_idx::Int)
    @assert b_idx != 1
    opt_a, opt_ba_idx = select_nonroot_action(planner, b_idx)
    mcts_main(planner, b_idx, opt_a, opt_ba_idx)
end

function mcts_main(planner::GumbelPlanner, b_idx::Int, a, ba_idx::Int)
    (; tree, pomdp, sol) = planner
    (; k_o, alpha_o, rng, check_repeat_obs) = sol

    if isterminalbelief(tree.b[b_idx])
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
            @assert b_idx != 1
            push!(tree.ba_children[ba_idx], bp_idx)
            Vp = mcts_main(planner, bp_idx)
        else
            bp_idx, Vp = insert_new_belief!(planner, b, ba_idx, a, o, p_idx, sample_sp, sample_r)
            @assert bp_idx != 1
        end
    else
        bp_idx = argmin(_bp_idx -> tree.Nh[_bp_idx], tree.ba_children[ba_idx])
        Vp = mcts_main(planner, bp_idx)
    end

    total = tree.b_rewards[bp_idx] + discount(pomdp) * tree.b_ntprob[bp_idx] * Vp

    # update tree
    tree.Nh[b_idx]   += 1
    tree.Nha[ba_idx] += 1
    tree.Qha[ba_idx] += (total - tree.Qha[ba_idx]) / tree.Nha[ba_idx]

    update_dq!(planner, tree.Qha[ba_idx])

    # return sum of rewards
    return total::Float64
end

function insert_new_belief!(planner, b, ba_idx, a, o, p_idx, sample_sp, sample_r)
    (; tree, pomdp, sol, cache) = planner
    (; rng, check_repeat_obs, resample, getpolicyvalue) = sol

    bp_particles, bp_weights = gen_empty_belief(cache, n_particles(b))

    bp, r, nt_prob = GenBelief(
        rng, bp_particles, bp_weights, cache.resample, 
        pomdp, b, a, o, p_idx, sample_sp, sample_r, resample
    )

    (; value, policy) = getpolicyvalue(bp)

    bp_idx = insert_belief!(tree, bp, ba_idx, o, r, nt_prob, value, policy, check_repeat_obs)
    
    update_dq!(planner, value)

    return bp_idx, value
end

function initialize_root!(planner::GumbelPlanner, particle_b)
    (; tree, sol, live_actions, noisy_logits) = planner
    (; rng, getpolicyvalue, m_acts_init) = sol

    pv = getpolicyvalue(particle_b)
    insert_root!(tree, particle_b, pv.value, pv.policy)

    # add gumbel noise to logits for categorical sampling
    # store noisy logits for later use in action selection
    gumbel!(rng, noisy_logits)
    map!(+, noisy_logits, noisy_logits, pv.policy)

    # sample `m_acts_init` actions without replacement and insert them into the tree
    fill!(live_actions, false)
    for ai in partialsortperm(noisy_logits, 1:m_acts_init; rev=true)
        live_actions[ai] = true
        insert_action!(tree, 1, ai)
    end
    
    return nothing
end

function gumbel!(rng, x) # in place sampling of Gumbel random variables
    rand!(rng, x)
    map!(y -> -log(-log(y)), x, x)
    return x
end

function select_root_action(planner::GumbelPlanner)
    (; tree, live_actions, target_N) = planner

    halving_flag = true
    for (ai, ba_idx) in tree.b_children[1]
        if live_actions[ai] && tree.Nha[ba_idx] < target_N.N
            halving_flag = false
            break
        end
    end

    if halving_flag
        next!(target_N)
        if count(live_actions) > 2
            reduce_root_actions(planner)
        end
    end

    return _select_root_action(tree, live_actions)
end

function _select_root_action(tree::GuidedTree, live_actions::AbstractVector{Bool})
    Nmax = typemax(Int)
    ai_opt = 0
    ba_idx_opt = 0
    for (ai, ba_idx) in tree.b_children[1]
        if live_actions[ai] && tree.Nha[ba_idx] < Nmax
            Nmax = tree.Nha[ba_idx]
            ai_opt = ai
            ba_idx_opt = ba_idx
        end
    end
    return (ai_opt, ba_idx_opt)
end

function reduce_root_actions(planner::GumbelPlanner)
    (; tree, sol, live_actions, noisy_logits) = planner
    (; cscale, cvisit) = sol

    dq = get_dq(planner)
    Nmax = iszero(tree.Nh[1]) ? 0 : maximum(tree.Nha[ba_idx] for (_, ba_idx) in tree.b_children[1])
    sigma = cscale * (cvisit + Nmax) / dq

    for _ in 1:floor(Int, count(live_actions) / 2)
        aidx_min = 0
        valmin = Inf
        for (ai, ba_idx) in tree.b_children[1]
            !live_actions[ai] && continue
            val = noisy_logits[ai] + sigma * tree.Qha[ba_idx]
            if val < valmin
                valmin = val
                aidx_min = ai
            end
        end

        live_actions[aidx_min] = false
    end

    nothing
end

function select_nonroot_action(planner::GumbelPlanner, b_idx::Int)
    (; tree, ordered_actions) = planner

    pi_completed = improved_policy(planner, b_idx)

    max_target = pi_completed
    for (ai, ba_idx) in tree.b_children[b_idx]
        max_target[ai] += tree.Nha[ba_idx] / (1 + tree.Nh[b_idx])
    end

    opt_ai = argmax(max_target)

    opt_a = ordered_actions[opt_ai]
    opt_ba_idx = insert_action!(tree, b_idx, opt_ai)

    return opt_a, opt_ba_idx
end

function improved_policy(planner::GumbelPlanner, b_idx::Int)
    (; tree, sol) = planner
    (; cscale, cvisit) = sol

    Nmax = maximum(tree.Nha[ba_idx] for (_, ba_idx) in tree.b_children[b_idx]; init = 0)
    dq = get_dq(planner)
    sigma = cscale * (cvisit + Nmax) / dq

    v_mix = get_v_mix(tree, b_idx)

    new_logits = tree.b_logits[b_idx] .+ sigma * v_mix
    for (ai, ba_idx) in tree.b_children[b_idx]
        new_logits[ai] += sigma * (tree.Qha[ba_idx] - v_mix)
    end

    pi_completed = softmax!(new_logits)

    return pi_completed
end

function get_v_mix(tree::GuidedTree, b_idx::Int)
    (; b_P, b_V, b_children, Qha, Nh) = tree

    P = b_P[b_idx]

    sum_pi = zero(eltype(P))
    sum_pi_q = zero(promote_type(eltype(Qha), eltype(P)))
    for (ai, ba_idx) in b_children[b_idx]
        sum_pi += P[ai]
        sum_pi_q += P[ai] * Qha[ba_idx]
    end

    v_mix = (b_V[b_idx] + Nh[b_idx] / sum_pi * sum_pi_q) / (1 + Nh[b_idx])

    return v_mix
end