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
    beliefcache_size::Int   = 1 + tree_queries
    treecache_size::Int     = 1 + tree_queries
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
    q_extrema::Vector{Float64}
    noisy_logits::Vector{Float64}
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

function POMDPs.solve(sol::GumbelSolver, pomdp::POMDP{S,A,O}) where {S,A,O}
    cache = BeliefCache{S}(min(sol.tree_queries, sol.beliefcache_size), sol.n_particles)

    na = length(actions(pomdp))

    tree = GuidedTree{S,Int,O}(
        min(sol.tree_queries, sol.treecache_size),
        na,
        sol.check_repeat_obs, 
        sol.k_o
    )

    ordered_actions = POMDPTools.ordered_actions(pomdp)
    live_actions = falses(na)

    q_extrema = [Inf, -Inf] # (min, max)

    noisy_logits = zeros(na)

    return GumbelPlanner(pomdp, sol, tree, cache, ordered_actions, live_actions, q_extrema, noisy_logits)
end

POMDPs.action(planner::GumbelPlanner, b) = first(POMDPTools.action_info(planner, b))
function POMDPTools.action_info(planner::GumbelPlanner, b)
    t0 = time()

    (; tree, cache, sol, pomdp, ordered_actions, live_actions) = planner
    (; rng, n_particles, default_action) = sol

    free!(cache)

    s, w = gen_empty_belief(cache, n_particles)
    particle_b = initialize_belief!(rng, s, w, pomdp, b)

    n_iter = _GumbelMCTS_main(planner, t0, particle_b)

    if isempty(first(tree.b_children))
        @warn "Taking random action"
        a = default_action(pomdp, b)
    else
        idxs = findall(live_actions)
        if length(idxs) == 1
            a = ordered_actions[idxs[1]]
        else
            @warn "Multiple live actions found. This functionality is not yet implemented"
            a = ordered_actions[idxs[1]]
        end 
    end

    Q_root = Dict(ordered_actions[ai]=>tree.Qha[ba_idx] for (ai, ba_idx) in tree.b_children[1])
    N_root = Dict(ordered_actions[ai]=>tree.Nha[ba_idx] for (ai, ba_idx) in tree.b_children[1])

    time = time() - to

    info = (n_iter, tree, time, Q_root, N_root)

    return a, info
end

function _GumbelMCTS_main(planner::GumbelPlanner, t0, particle_b)
    (; sol, live_actions) = planner
    (; max_time, tree_queries, m_acts_init) = sol

    initialize_root!(planner, particle_b)

    dN = tree_queries/(ceil(log2(m_acts_init)) * nextpow(2,m_acts_init))
    ktarget = 1
    Ntarget = floor(Int, dN)

    t_max = t0 + max_time
    for iter in 1:tree_queries
        (time() >= t_max) && return iter - 1

        sra = select_root_action(planner, Ntarget)
        if isnothing(sra)
            ktarget *= 2
            Ntarget += floor(Int, ktarget * dN)

            reduce_root_actions(planner)
            @assert count(live_actions) > 0
            count(live_actions) == 1 && break

            sra = select_root_action(planner, Ntarget)
        end
        a, ba_idx = sra

        mcts_main(planner, 1, a, ba_idx, 0)
    end

    return tree_queries
end

function initialize_root!(planner::GumbelPlanner, particle_b)
    (; tree, sol, live_actions, noisy_logits) = planner
    (; rng, getpolicyvalue, m_acts_init, stochastic_root) = sol

    pv = getpolicyvalue(particle_b)
    P = pv.policy
    V = pv.value
    insert_root!(tree, particle_b, V, P)


    if stochastic_root
        # gumbel distribution
        rand!(rng, noisy_logits)
        map!(x -> -log(-log(x)), noisy_logits, noisy_logits)
    else
        fill!(noisy_logits, zero(eltype(noisy_logits)))
    end
    map!(+, noisy_logits, noisy_logits, P) # zero allocations this way

    fill!(live_actions, false)
    top_a_idx = sortperm(noisy_logits; rev=true) # allocation :(
    for ai in top_a_idx[1:m_acts_init]
        live_actions[ai] = true
        insert_action!(tree, 1, ai)
    end
    
    return nothing
end

function select_root_action(planner::GumbelPlanner, Ntarget)
    (; tree, ordered_actions, live_actions) = planner

    # select action with minimum Nha - constrained by Ntarget and live_actions
    # right now this only works if halving ends within iteration bounds

    for (ai, ba_idx) in tree.b_children[1]
        if live_actions[ai] && tree.Nha[ba_idx] < Ntarget
            a = ordered_actions[ai]
            return (a, ba_idx)
        end
    end

    return nothing
end

function reduce_root_actions(planner::GumbelPlanner)
    (; tree, sol, live_actions, noisy_logits) = planner
    (; cscale, cvisit) = sol

    dq = get_dq(planner)
    Nmax = iszero(tree.Nh[1]) ? 0 : maximum(tree.Nha[ba_idx] for (_, ba_idx) in tree.b_children[1])
    sigma = cscale * (cvisit + Nmax) / dq

    for _ in 1:floor(Int, count(live_actions)/2)
        local aidx_min::Int
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

function mcts_main(planner::GumbelPlanner, b_idx::Int, d::Int)
    (; tree, sol, ordered_actions) = planner
    (; cscale, cvisit) = sol

    Nmax = iszero(tree.Nh[b_idx]) ? 0 : maximum(tree.Nha[ba_idx] for (_, ba_idx) in tree.b_children[b_idx])
    dq = get_dq(planner)
    sigma = cscale * (cvisit + Nmax) / dq

    P = softmax(tree.b_P[b_idx])
    sum_pi = sum(P[ai] for (ai, _) in tree.b_children[b_idx]; init=zero(eltype(P)))
    sum_pi_q = sum(P[ai] * tree.Qha[ba_idx] for (ai,ba_idx) in tree.b_children[b_idx]; init=zero(eltype(tree.Qha)))
    v_mix = (tree.b_V[b_idx] + tree.Nh[b_idx] / sum_pi * sum_pi_q) / (1 + tree.Nh[b_idx])

    new_logits = tree.b_P[b_idx] .+ sigma * v_mix
    for (ai, ba_idx) in tree.b_children[b_idx]
        new_logits[ai] += sigma * (tree.Qha[ba_idx] - v_mix)
    end
    pi_completed = softmax!(new_logits)

    for (ai, ba_idx) in tree.b_children[b_idx]
        pi_completed[ai] -= tree.Nha[ba_idx] / (1 + tree.Nh[b_idx])
    end

    opt_ai = argmax(pi_completed)
    opt_a = ordered_actions[opt_ai]
    opt_ba_idx = insert_action!(tree, b_idx, opt_ai)

    mcts_main(planner, b_idx, opt_a, opt_ba_idx, d)
end

function mcts_main(planner::GumbelPlanner, b_idx::Int, a, ba_idx::Int, d::Int)
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
            Vp = mcts_main(planner, bp_idx, d+1)
        else
            bp_idx, Vp = insert_new_belief!(planner, b, ba_idx, a, o, p_idx, sample_sp, sample_r)
        end
    else
        bp_idx = rand(rng, tree.ba_children[ba_idx])
        @assert !isempty(tree.ba_children[ba_idx]) "depth = $d"
        Vp = mcts_main(planner, bp_idx, d+1)
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
