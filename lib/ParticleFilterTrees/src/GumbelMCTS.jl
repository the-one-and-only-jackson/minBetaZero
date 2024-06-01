"""
...
- `max_depth::Int = 20` - Maximum tree search depth
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
    max_depth::Int          = 20
    n_particles::Int        = 100
    k_o::Float64            = 10.0
    alpha_o::Float64        = 0.0 # Observation Progressive widening parameter
    rng::RNG                = Random.default_rng()
    check_repeat_obs::Bool  = true
    beliefcache_size::Int   = 1_000
    treecache_size::Int     = 1_000
    default_action::DA      = (pomdp::POMDP, ::Any) -> rand(rng, actions(pomdp))
    resample::Bool          = true
    cscale::Float64         = 1.
    cvisit::Float64         = 50.
    stochastic_root::Bool   = false
    m_acts_init::Int
    getpolicyvalue::F
end

struct GumbelPlanner{SOL<:GumbelSolver, M<:POMDP, TREE<:GuidedTree, S, OA} <: Policy
    pomdp::M
    sol::SOL
    tree::TREE
    cache::BeliefCache{S}
    ordered_actions::OA
    live_actions::BitVector
end

function POMDPs.solve(sol::GumbelSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    cache = BeliefCache{S}(min(sol.tree_queries, sol.beliefcache_size), sol.n_particles)

    tree = GuidedTree{S,Int,O}(
        min(sol.tree_queries, sol.treecache_size),
        length(actions(pomdp)),
        sol.check_repeat_obs, 
        sol.k_o
    )

    ordered_actions = POMDPTools.ordered_actions(pomdp)
    live_actions = falses(length(ordered_actions))

    return GumbelPlanner(pomdp, sol, tree, cache, ordered_actions, live_actions)
end

POMDPs.action(planner::GumbelPlanner, b) = first(POMDPTools.action_info(planner, b))
function POMDPTools.action_info(planner::GumbelPlanner, b)
    t0 = time()

    (; tree, cache, sol, pomdp, ordered_actions, live_actions) = planner
    (; max_time, tree_queries, default_action, m_acts_init) = sol

    free!(cache)

    g_P = initialize_root!(planner, b)
    dN = tree_queries/(ceil(log2(m_acts_init)) * nextpow(2,m_acts_init))
    Ntarget = floor(Int, dN)
    ktarget = 1

    iter = 0
    while (time()-t0 < max_time) && (iter < tree_queries)
        sra = select_root_action(planner, Ntarget)
        if isnothing(sra)
            ktarget *= 2
            Ntarget += floor(Int, ktarget * dN)

            reduce_root_actions(planner, Ntarget, g_P)
            @assert count(live_actions) > 0
            count(live_actions) == 1 && break

            sra = select_root_action(planner, Ntarget)
        end
        a, ba_idx = sra

        mcts_main(planner, 1, a, ba_idx, 0, Ntarget)

        iter += 1
    end

    if isempty(first(tree.b_children))
        @warn "Taking random action"
        a = default_action(pomdp, b)
    else
        a = ordered_actions[findfirst(live_actions)]
    end

    info = (
        n_iter = iter,
        tree   = tree,
        time   = time() - t0,
        Q_root = Dict(ordered_actions[ai]=>tree.Qha[ba_idx] for (ai, ba_idx) in tree.b_children[1]),
        N_root = Dict(ordered_actions[ai]=>tree.Nha[ba_idx] for (ai, ba_idx) in tree.b_children[1])
    )

    return a, info
end

function initialize_root!(planner, b)
    (; tree, cache, sol, pomdp, live_actions) = planner
    (; rng, n_particles, tree_queries, getpolicyvalue, m_acts_init, stochastic_root) = sol

    s,w = gen_empty_belief(cache, n_particles)
    particle_b = initialize_belief!(rng, s, w, pomdp, b)

    na = length(actions(pomdp))

    pv = getpolicyvalue(particle_b)
    P = pv.policy
    V = pv.value
    insert_root!(tree, particle_b, V, P)

    noise = -log.(-log.(rand(rng, na)))
    g_P = stochastic_root ? P + noise : P

    fill!(live_actions, false)
    top_a_idx = sortperm(g_P; rev=true)
    for i in 1:na
        i > m_acts_init && continue
        ai = top_a_idx[i]
        live_actions[ai] = true
        insert_action!(tree, 1, ai)
    end
    
    return g_P
end

function select_root_action(planner::GumbelPlanner, Ntarget)
    (; tree, pomdp, ordered_actions, live_actions) = planner

    for (ai, ba_idx) in tree.b_children[1]
        if live_actions[ai] && tree.Nha[ba_idx] < Ntarget
            a = ordered_actions[ai]
            return (a, ba_idx)
        end
    end

    return nothing
end

function get_q_extrema(tree::GuidedTree, b_idx::Int)
    min_q = max_q = Float64(tree.b_V[b_idx])
    for (ai, ba_idx) in tree.b_children[b_idx]
        q = tree.Qha[ba_idx]
        if q < min_q
            min_q = q
        elseif q > max_q
            max_q = q
        end
    end
    return min_q, max_q
end

function reduce_root_actions(planner::GumbelPlanner, Ntarget::Int, g_P)
    (; tree, sol, pomdp, live_actions) = planner
    (; cscale, cvisit) = sol

    min_q, max_q = get_q_extrema(tree, 1)
    dq = max_q - min_q

    sigma = cscale * (cvisit + Ntarget) / dq

    for i in 1:floor(Int, count(live_actions)/2)
        local aidx_min::Int
        valmin = Inf
        for (ai, ba_idx) in tree.b_children[1]
            !live_actions[ai] && continue
            val = g_P[ai] + sigma * tree.Qha[ba_idx]
            if val < valmin
                valmin = val
                aidx_min = ai
            end
        end
        live_actions[aidx_min] = false
    end
    nothing
end

function mcts_main(planner::GumbelPlanner, b_idx::Int, d::Int, Ntarget::Int)
    (; tree, pomdp, sol, ordered_actions) = planner
    (; cscale, cvisit) = sol

    sigma = cscale * (cvisit + Ntarget)
    v = tree.b_V[b_idx]

    min_q, max_q = get_q_extrema(tree, b_idx)
    dq = max_q - min_q

    sigma_v_norm = sigma * v / dq

    new_logits = tree.b_P[b_idx] .+ sigma_v_norm # add a cache here!!!
    for (ai, ba_idx) in tree.b_children[b_idx]
        new_logits[ai] += sigma * tree.Qha[ba_idx] / dq - sigma_v_norm
    end
    pi_completed = softmax!(new_logits)

    Nh_1 = 1 + tree.Nh[b_idx]
    for (ai, ba_idx) in tree.b_children[b_idx]
        pi_completed[ai] -= tree.Nha[ba_idx]/Nh_1
    end

    opt_ai = argmax(pi_completed)
    opt_a = ordered_actions[opt_ai]
    opt_ba_idx = insert_action!(tree, b_idx, opt_ai)

    mcts_main(planner, b_idx, opt_a, opt_ba_idx, d, Ntarget)
end

function mcts_main(planner::GumbelPlanner, b_idx::Int, a, ba_idx::Int, d::Int, Ntarget::Int)
    (; tree, pomdp, sol) = planner
    (; max_depth, k_o, alpha_o, rng, check_repeat_obs) = sol

    if d==max_depth || isterminalbelief(tree.b[b_idx])
        return tree.b_V[b_idx]
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
        @assert !isempty(tree.ba_children[ba_idx]) "depth = $d"
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
    (; tree, pomdp, sol, cache, ordered_actions) = planner
    (; max_depth, rng, check_repeat_obs, resample, getpolicyvalue) = sol

    bp_particles, bp_weights = gen_empty_belief(cache, n_particles(b))
    bp, r, nt_prob = GenBelief(
        rng, bp_particles, bp_weights, cache.resample, 
        pomdp, b, a, o, p_idx, sample_sp, sample_r, resample
    )

    (; value, policy) = getpolicyvalue(bp)

    bp_idx = insert_belief!(tree, bp, ba_idx, o, r, nt_prob, value, policy, check_repeat_obs)

    return bp_idx, value
end
