Base.@kwdef struct GumbelSolver{RNG<:AbstractRNG, DA, F} <: Solver
    tree_queries::Int       = 1_000
    max_time::Float64       = Inf # (seconds)
    n_particles::Int        = 100
    k_o::Float64            = 10.0
    alpha_o::Float64        = 0.0 # Observation Progressive widening parameter
    rng::RNG                = Random.default_rng()
    default_action::DA      = (pomdp::POMDP, ::Any) -> rand(rng, actions(pomdp))
    resample::Bool          = true
    cscale::Float64         = 0.1 # 1 for win/loss games, 0.1 for MDP
    cvisit::Float64         = 50.
    m_acts_init::Int        = typemax(Int)
    getpolicyvalue::F
end

@kwdef struct GumbelPlanner{SOL<:GumbelSolver, M<:POMDP, TREE<:GuidedTree, S, OA} <: Policy
    pomdp::M
    sol::SOL
    tree::TREE
    cache::BeliefCache{S}
    ordered_actions::OA = POMDPTools.ordered_actions(pomdp)
    live_actions::BitVector = falses(length(actions(pomdp)))
    q_extrema::Vector{Float64} = [Inf, -Inf]
    noisy_logits::Vector{Float64} = zeros(length(actions(pomdp)))
    target_N::SeqHalf = SeqHalf(; n=sol.tree_queries, m=min(sol.m_acts_init, length(actions(pomdp))))
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

function POMDPs.solve(sol::GumbelSolver, pomdp::POMDP{S,A}) where {S,A}
    (; tree_queries, n_particles, k_o) = sol
    cache = BeliefCache{S}(tree_queries + 1, n_particles)
    tree = GuidedTree{PFTBelief{S},Int}(tree_queries + 1, length(actions(pomdp)), k_o)
    return GumbelPlanner(; pomdp, sol, tree, cache)
end

POMDPs.action(planner::GumbelPlanner, b) = first(POMDPTools.action_info(planner, b))

function POMDPTools.action_info(planner::GumbelPlanner, b_root)
    (; tree, cache, sol, pomdp, ordered_actions, target_N) = planner
    (; rng, n_particles, default_action, max_time, tree_queries, getpolicyvalue) = sol
    (; Qha, Nha, b_children) = tree

    t0 = time()

    free!(cache)

    s, w = gen_empty_belief(cache, n_particles)
    particle_b_root = initialize_belief!(rng, s, w, pomdp, b_root)

    if isterminalbelief(particle_b_root)
        a = default_action(pomdp, b_root)
        n_iter = 0
        tree = nothing
        Q_root = nothing
        N_root = nothing
        policy_target = nothing
    else
        reset_tree!(tree)
        reset!(target_N)

        n_iter = tree_queries
        for iter in 1:tree_queries
            if time() >= t0 + max_time
                n_iter = iter - 1
                break
            end

            b_querry, ba_idx, r, done = mcts_forward(planner, particle_b_root)

            if done
                value, policy = 0f0, zeros(Float32, length(ordered_actions))
            else
                (; value, policy) = getpolicyvalue(b_querry)
            end

            mcts_backward!(planner, b_querry, ba_idx, r, value, policy)
        end

        a = select_best_action(planner)

        Q_root = Dict(ordered_actions[ai] => Qha[ba_idx] for (ai, ba_idx) in b_children[1])
        N_root = Dict(ordered_actions[ai] => Nha[ba_idx] for (ai, ba_idx) in b_children[1])

        policy_target, v_mix = improved_policy(planner, 1)
    end

    return a, (; n_iter, tree, time = time() - t0, Q_root, N_root, policy_target, v_mix)
end

function select_best_action(planner::GumbelPlanner)
    (; tree, sol, live_actions, noisy_logits, ordered_actions) = planner
    (; cscale, cvisit) = sol
    (; Nh, Nha, Qha, b_children) = tree

    if count(live_actions) == 1
        ai_opt = findfirst(live_actions)
    else
        dq = get_dq(planner)
        Nmax = iszero(Nh[1]) ? 0 : maximum(Nha[ba_idx] for (_, ba_idx) in b_children[1])
        sigma = cscale * (cvisit + Nmax) / dq

        ai_opt = 0
        valmax = -Inf
        for (ai, ba_idx) in b_children[1]
            !live_actions[ai] && continue
            val = noisy_logits[ai] + sigma * Qha[ba_idx]
            if val > valmax
                valmax = val
                ai_opt = ai
            end
        end
    end

    return ordered_actions[ai_opt]
end

function mcts_forward(planner::GumbelPlanner, b_root)
    (; tree, sol, ordered_actions) = planner
    (; k_o, alpha_o) = sol
    (; Nh, Nha, b, ba_children, b_children) = tree

    if isempty(b_children)
        b_querry = b_root
        ba_idx = 0
        r = 0.0
        done = false
    else
        # do root stuff
        b_idx = 1
        ai, ba_idx = select_root_action(planner)
        a = ordered_actions[ai]
        b_querry = b[b_idx]
        r = 0.0
        done = false

        # do nonroot stuff
        depth = 0
        while true
            depth += 1
            @assert depth < 1 + length(b) "Loop has spiraled out of control!"

            if isterminalbelief(b[b_idx])
                b_querry = b[b_idx]
                done = true
                break
            elseif length(ba_children[ba_idx]) < k_o * Nha[ba_idx] ^ alpha_o
                b_querry, r, done = gen_querry(planner, tree.b[b_idx], a)
                break
            else
                b_idx = argmin(_bp_idx -> Nh[_bp_idx], ba_children[ba_idx])
                a, ba_idx = select_nonroot_action(planner, b_idx)
            end
        end
    end

    return b_querry, ba_idx, r, done
end

function gen_querry(planner::GumbelPlanner, b, a)
    (; pomdp, sol, cache) = planner
    (; rng, resample) = sol

    p_idx = non_terminal_sample(rng, pomdp, b)
    sample_s = particle(b, p_idx)
    sample_sp, o, sample_r = @gen(:sp,:o,:r)(pomdp, sample_s, a, rng)

    bp_particles, bp_weights = gen_empty_belief(cache, n_particles(b))

    b_querry, r, _ = GenBelief(
        rng, bp_particles, bp_weights, cache.resample,
        pomdp, b, a, o, p_idx, sample_sp, sample_r, resample
    )

    done = isterminalbelief(b_querry)

    return b_querry, r, done
end

function mcts_backward!(planner::GumbelPlanner, b_querry, ba_idx, r, value, logits)
    update_dq!(planner, value)

    b_idx = insert_belief!(planner.tree, b_querry; ba_idx, r, value, logits)

    if b_idx == 1
        mcts_backward_root!(planner, logits)
    else
        mcts_backward_nonroot!(planner, b_idx, ba_idx, value)
    end

    return nothing
end

function mcts_backward_root!(planner::GumbelPlanner, logits)
    (; tree, sol, live_actions, noisy_logits) = planner
    (; rng, m_acts_init) = sol

    # add gumbel noise to logits for categorical sampling
    # store noisy logits for later use in action selection
    rand!(rng, noisy_logits)
    map!(y -> -log(-log(y)), noisy_logits, noisy_logits)
    map!(+, noisy_logits, noisy_logits, logits)

    # sample `n_root_actions` actions without replacement and insert them into the tree
    n_root_actions = min(m_acts_init, length(live_actions))
    fill!(live_actions, false)
    for ai in partialsortperm(noisy_logits, 1:n_root_actions; rev=true)
        live_actions[ai] = true
        insert_action!(tree, 1, ai)
    end

    return nothing
end

function mcts_backward_nonroot!(planner::GumbelPlanner, b_idx::Int, ba_idx::Int, value::Real)
    (; tree, pomdp) = planner
    (; b_rewards, b_parent, ba_parent, Nh, Nha, Qha) = tree

    gamma = discount(pomdp)

    while !iszero(ba_idx)
        value = b_rewards[b_idx] + gamma * value

        Nha[ba_idx] += 1
        Qha[ba_idx] += (value - Qha[ba_idx]) / Nha[ba_idx]

        update_dq!(planner, Qha[ba_idx])

        b_idx = b_parent[ba_idx]
        ba_idx = ba_parent[b_idx]

        Nh[b_idx] += 1
    end

    return nothing
end

function select_root_action(planner::GumbelPlanner)
    (; tree, live_actions, target_N) = planner
    (; Nha, b_children) = tree

    halving_flag = true

    for (ai, ba_idx) in b_children[1]
        if live_actions[ai] && Nha[ba_idx] < target_N.N
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

    # Not sure why this errors out
    # ai, ba_idx = argmin(
    #     (ai, ba_idx) -> Nha[ba_idx],
    #     ((ai, ba_idx) for (ai, ba_idx) in b_children[1] if live_actions[ai])
    # )

    Nmax = typemax(Int)
    ai_opt = 0
    ba_idx_opt = 0
    for (ai, ba_idx) in b_children[1]
        if live_actions[ai] && Nha[ba_idx] < Nmax
            Nmax = Nha[ba_idx]
            ai_opt = ai
            ba_idx_opt = ba_idx
        end
    end

    return ai_opt, ba_idx_opt
end

function reduce_root_actions(planner::GumbelPlanner)
    (; tree, sol, live_actions, noisy_logits) = planner
    (; cscale, cvisit) = sol
    (; Nh, Nha, Qha, b_children) = tree

    dq = get_dq(planner)
    Nmax = iszero(Nh[1]) ? 0 : maximum(Nha[ba_idx] for (_, ba_idx) in b_children[1])
    sigma = cscale * (cvisit + Nmax) / dq

    for _ in 1:floor(Int, count(live_actions) / 2)
        ai, _ = argmin(
            (ai, ba_idx) -> noisy_logits[ai] + sigma * Qha[ba_idx],
            ((ai, ba_idx) for (ai, ba_idx) in b_children[1] if live_actions[ai])
        )

        live_actions[ai] = false
    end

    nothing
end

function select_nonroot_action(planner::GumbelPlanner, b_idx::Int)
    (; tree, ordered_actions) = planner
    (; Nh, Nha, b_children) = tree

    pi_completed, _ = improved_policy(planner, b_idx)

    max_target = pi_completed
    for (ai, ba_idx) in b_children[b_idx]
        max_target[ai] += Nha[ba_idx] / (1 + Nh[b_idx])
    end

    opt_ai = argmax(max_target)

    opt_a = ordered_actions[opt_ai]
    opt_ba_idx = insert_action!(tree, b_idx, opt_ai)

    return opt_a, opt_ba_idx
end

function improved_policy(planner::GumbelPlanner, b_idx::Int)
    (; tree, sol) = planner
    (; cscale, cvisit) = sol
    (; Nha, Qha, b_children, b_logits) = tree

    Nmax = maximum(Nha[ba_idx] for (_, ba_idx) in b_children[b_idx]; init = 0)
    dq = get_dq(planner)
    sigma = cscale * (cvisit + Nmax) / dq

    v_mix = get_v_mix(tree, b_idx)

    new_logits = b_logits[b_idx] .+ sigma * v_mix
    for (ai, ba_idx) in b_children[b_idx]
        advantage = Qha[ba_idx] - v_mix
        new_logits[ai] += sigma * advantage
    end

    pi_completed = softmax!(new_logits)

    return pi_completed, v_mix
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
