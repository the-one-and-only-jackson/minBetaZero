Base.@kwdef struct GumbelSolver{RNG<:AbstractRNG, DA, F} <: Solver
    tree_queries::Int       = 1_000
    max_time::Float64       = Inf # (seconds)
    k_o::Float64            = 10.0
    alpha_o::Float64        = 0.0 # Observation Progressive widening parameter
    rng::RNG                = Random.default_rng()
    default_action::DA      = (mdp::MDP, ::Any) -> rand(rng, actions(mdp))
    resample::Bool          = true
    cscale::Float64         = 1.
    cvisit::Float64         = 50.
    m_acts_init::Int
    getpolicyvalue::F
end

@kwdef struct GumbelPlanner{SOL<:GumbelSolver, M<:MDP, TREE<:GuidedTree, S, OA} <: Policy
    mdp::M
    sol::SOL
    tree::TREE
    ordered_actions::OA = POMDPTools.ordered_actions(mdp)
    live_actions::BitVector = falses(length(actions(mdp)))
    q_extrema::Vector{Float64} = [Inf, -Inf]
    noisy_logits::Vector{Float64} = zeros(length(actions(mdp)))
    target_N::SeqHalf = SeqHalf(; n=sol.tree_queries, m=sol.m_acts_init)
end

function update_dq!(planner::GumbelPlanner, q)
    if q < planner.q_extrema[1]
        planner.q_extrema[1] = q
    elseif q > planner.q_extrema[2]
        planner.q_extrema[2] = q
    end
    return nothing
end

function get_sigma(planner::GumbelPlanner, s_idx::Int; eps=1e-6)
    (; sol, tree) = planner
    (; cscale, cvisit) = sol
    (; Nha, s_children) = tree

    Nmax = maximum(Nha[sa_idx] for (_, sa_idx) in s_children[s_idx]; init = 0)
    sigma = cscale * (cvisit + Nmax)

    dq = planner.q_extrema[2] - planner.q_extrema[1]
    if dq > eps
        sigma /= dq
    end

    return sigma
end

function POMDPs.solve(sol::GumbelSolver, mdp::MDP{S,A}) where {S,A}
    (; tree_queries, k_o) = sol
    tree = GuidedTree{S,Int}(tree_queries + 1, length(actions(mdp)), k_o)
    return GumbelPlanner(; mdp, sol, tree)
end

POMDPs.action(planner::GumbelPlanner, s) = first(POMDPTools.action_info(planner, s))

function POMDPTools.action_info(planner::GumbelPlanner, s_root)
    (; tree, cache, sol, mdp, ordered_actions, target_N) = planner
    (; default_action, max_time, tree_queries, getpolicyvalue) = sol
    (; Qha, Nha, s_children) = tree

    t0 = time()

    if isterminal(mdp, s_root)
        a = default_action(mdp, s_root)
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

            s_querry, sa_idx, r, done = mcts_forward(planner, s_root)

            if done
                value, policy = 0f0, zeros(Float32, length(ordered_actions))
            else
                (; value, policy) = getpolicyvalue(s_querry)
            end

            mcts_backward!(planner, s_querry, sa_idx, r, value, policy)
        end

        a = select_best_action(planner)

        Q_root = Dict(ordered_actions[ai] => Qha[sa_idx] for (ai, sa_idx) in s_children[1])
        N_root = Dict(ordered_actions[ai] => Nha[sa_idx] for (ai, sa_idx) in s_children[1])

        policy_target, v_mix = improved_policy(planner, 1)
    end

    return a, (; n_iter, tree, time = time() - t0, Q_root, N_root, policy_target, v_mix)
end

function select_best_action(planner::GumbelPlanner)
    (; tree, live_actions, noisy_logits, ordered_actions) = planner
    (; Qha, s_children) = tree

    if count(live_actions) == 1
        ai_opt = findfirst(live_actions)
    else
        sigma = get_sigma(planner, 1)

        ai_opt, _ = argmax(
            (ai, sa_idx) -> noisy_logits[ai] + sigma * Qha[sa_idx],
            ((ai, sa_idx) for (ai, sa_idx) in s_children[1] if live_actions[ai])
        )
    end

    return ordered_actions[ai_opt]
end

function mcts_forward(planner::GumbelPlanner, b_root)
    (; tree, sol, ordered_actions) = planner
    (; k_o, alpha_o) = sol
    (; Nh, Nha, s, sa_children, s_children) = tree

    if isempty(s_children)
        s_querry = b_root
        sa_idx = 0
        r = 0.0
        done = false
    else
        # do root stuff
        s_idx = 1
        ai, sa_idx = select_root_action(planner)
        a = ordered_actions[ai]
        s_querry = s[s_idx]
        r = 0.0
        done = false

        # do nonroot stuff
        depth = 0
        while true
            depth += 1
            @assert depth < 1 + length(s) "Loop has spiraled out of control!"

            if isterminalbelief(s[s_idx])
                s_querry = s[s_idx]
                done = true
                break
            elseif length(sa_children[sa_idx]) < k_o * Nha[sa_idx] ^ alpha_o
                s_querry, r, done = gen_querry(planner, s[s_idx], a)
                break
            else
                s_idx = argmin(_bp_idx -> Nh[_bp_idx], sa_children[sa_idx])
                a, sa_idx = select_nonroot_action(planner, s_idx)
            end
        end
    end

    return s_querry, sa_idx, r, done
end

function gen_querry(planner::GumbelPlanner, s, a)
    (; mdp, sol) = planner
    sp, r = @gen(:sp, :r)(mdp, s, a, sol.rng)
    done = isterminal(sp)
    return sp, r, done
end

function mcts_backward!(planner::GumbelPlanner, s_querry, sa_idx, r, value, logits)
    update_dq!(planner, value)

    s_idx = insert_belief!(planner.tree, s_querry; sa_idx, r, value, logits)

    if s_idx == 1
        mcts_backward_root!(planner, logits)
    else
        mcts_backward_nonroot!(planner, s_idx, sa_idx, value)
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

    # sample `m_acts_init` actions without replacement and insert them into the tree
    fill!(live_actions, false)
    for ai in partialsortperm(noisy_logits, 1:m_acts_init; rev=true)
        live_actions[ai] = true
        insert_action!(tree, 1, ai)
    end

    return nothing
end

function mcts_backward_nonroot!(planner::GumbelPlanner, s_idx::Int, sa_idx::Int, value::Real)
    (; tree, mdp) = planner
    (; b_rewards, s_Parent, ba_parent, Nh, Nha, Qha) = tree

    gamma = discount(mdp)

    while !iszero(sa_idx)
        value = b_rewards[s_idx] + gamma * value

        Nha[sa_idx] += 1
        Qha[sa_idx] += (value - Qha[sa_idx]) / Nha[sa_idx]

        update_dq!(planner, Qha[sa_idx])

        s_idx = s_Parent[sa_idx]
        sa_idx = ba_parent[s_idx]

        Nh[s_idx] += 1
    end

    return nothing
end

function select_root_action(planner::GumbelPlanner)
    (; tree, live_actions, target_N) = planner
    (; Nha, s_children) = tree

    halving_flag = true

    for (ai, sa_idx) in s_children[1]
        if live_actions[ai] && Nha[sa_idx] < target_N.N
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

    ai, sa_idx = argmin(
        (ai, sa_idx) -> Nha[sa_idx],
        ((ai, sa_idx) for (ai, sa_idx) in s_children[1] if live_actions[ai])
    )

    return ai, sa_idx
end

function reduce_root_actions(planner::GumbelPlanner)
    (; tree, live_actions, noisy_logits) = planner
    (; Qha, s_children) = tree

    sigma = get_sigma(planner, 1)

    for _ in 1:floor(Int, count(live_actions) / 2)
        ai, _ = argmin(
            (ai, sa_idx) -> noisy_logits[ai] + sigma * Qha[sa_idx],
            ((ai, sa_idx) for (ai, sa_idx) in s_children[1] if live_actions[ai])
        )
        live_actions[ai] = false
    end

    nothing
end

function select_nonroot_action(planner::GumbelPlanner, s_idx::Int)
    (; tree, ordered_actions) = planner
    (; Nh, Nha, s_children) = tree

    pi_completed, _ = improved_policy(planner, s_idx)

    max_target = pi_completed
    for (ai, sa_idx) in s_children[s_idx]
        max_target[ai] += Nha[sa_idx] / (1 + Nh[s_idx])
    end

    opt_ai = argmax(max_target)

    opt_a = ordered_actions[opt_ai]
    opt_sa_idx = insert_action!(tree, s_idx, opt_ai)

    return opt_a, opt_sa_idx
end

function improved_policy(planner::GumbelPlanner, s_idx::Int)
    (; Qha, s_children, prior_logits) = planner.tree

    sigma = get_sigma(planner, s_idx)
    v_mix = get_v_mix(tree, s_idx)

    new_logits = prior_logits[s_idx] .+ sigma * v_mix
    for (ai, sa_idx) in s_children[s_idx]
        new_logits[ai] += sigma * (Qha[sa_idx] - v_mix)
    end

    pi_completed = softmax!(new_logits)

    return pi_completed, v_mix
end

function get_v_mix(tree::GuidedTree, s_idx::Int)
    (; prior_policy, prior_value, s_children, Qha, Nh) = tree

    P = prior_policy[s_idx]

    sum_pi = zero(eltype(P))
    sum_pi_q = zero(promote_type(eltype(Qha), eltype(P)))
    for (ai, sa_idx) in s_children[s_idx]
        sum_pi += P[ai]
        sum_pi_q += P[ai] * Qha[sa_idx]
    end

    v_mix = (prior_value[s_idx] + Nh[s_idx] / sum_pi * sum_pi_q) / (1 + Nh[s_idx])

    return v_mix
end
