mutable struct GumbelSearch{M, S, A, RNG}
    mdp             :: MDP
    tree            :: GuidedTree{S, Int}
    tree_querries   :: Int
    ordered_actions :: Vector{A}
    m_acts_init     :: Int
    live_actions    :: BitVector
    target_N        :: SeqHalf
    k_o             :: Float64
    alpha_o         :: Float64
    q_extrema       :: Vector{Float64}
    cscale          :: Float64
    cvisit          :: Float64
    rng             :: RNG

    s_idx           :: Int

    function GumbelSearch(
            mdp :: M;
            tree_querries :: Real = length(actions(mdp)),
            m_acts_init   :: Real = length(actions(mdp)),
            k_o           :: Real = 1,
            alpha_o       :: Real = 0,
            cscale        :: Real = 0.1,
            cvisit        :: Real = 50,
            rng :: AbstractRNG = Random.default_rng()
        ) where {S, A, M <: MDP{S,A}}

        na = length(actions(mdp))

        tree_querries   = 1 + floor(Int, tree_querries) # root querry addition
        ordered_actions = POMDPTools.ordered_actions(mdp)
        m_acts_init     = floor(Int, m_acts_init)
        live_actions    = falses(na)
        target_N        = SeqHalf(; n=tree_querries, m=m_acts_init)

        k_o     = Float64(k_o)
        alpha_o = Float64(alpha_o)

        q_extrema = [Inf, -Inf]
        cscale    = Float64(cscale)
        cvisit    = Float64(cvisit)

        tree = GuidedTree{S, Int}(tree_queries + 1, na, ceil(Int, k_o))

        s_idx = 1

        return new(
            mdp,
            tree,
            tree_querries,
            ordered_actions,
            m_acts_init,
            live_actions,
            target_N,
            k_o,
            alpha_o,
            q_extrema,
            cscale,
            cvisit,
            rng,
            s_idx
        )
    end
end

function root_info(planner::GumbelPlanner)
    (; tree, ordered_actions) = planner
    (; Qha, Nha, s_children) = tree

    a = select_best_action(planner)

    Q_root = Dict(ordered_actions[ai] => Qha[sa_idx] for (ai, sa_idx) in s_children[1])
    N_root = Dict(ordered_actions[ai] => Nha[sa_idx] for (ai, sa_idx) in s_children[1])

    policy_target, v_mix = improved_policy(planner, 1)

    return a, (; tree, Q_root, N_root, policy_target, v_mix)
end

function select_best_action(planner::GumbelSearch)
    (; tree, live_actions, ordered_actions) = planner
    (; Qha, s_children) = tree

    root_logits = tree.prior_logits[1]

    sigma = get_sigma(planner, 1)

    ai_opt, _ = argmax(
        (ai, sa_idx) -> root_logits[ai] + sigma * Qha[sa_idx],
        ((ai, sa_idx) for (ai, sa_idx) in s_children[1] if live_actions[ai])
    )

    return ordered_actions[ai_opt]
end

Base.isdone(planner::GumbelSearch) = planner.tree.Nh[1] >= planner.tree_querries

function insert_root!(planner::GumbelSearch, s_root)
    (; tree, mdp, target_N, ordered_actions) = planner

    @assert !isterminal(mdp, s_root) "Root state is terminal! s = $s_root"

    reset_tree!(tree)
    reset!(target_N)

    insert_state!(planner.tree, s_root; logits = ones(Float32, length(ordered_actions)))

    return nothing
end

function mcts_forward!(planner::GumbelSearch)
    if isempty(planner.tree.s_children)
        s_idx = 1
        s_querry = planner.tree.s[s_idx]
    else
        s_querry, s_idx = mcts_forward_nonroot(planner)
    end
    planner.s_idx = s_idx
    return s_querry
end

function mcts_forward_nonroot!(planner::GumbelSearch)
    (; tree, ordered_actions, k_o, alpha_o, mdp, rng) = planner
    (; Nh, Nha, s, sa_children) = tree

    # do root stuff
    s_idx = 1
    ai, sa_idx = select_root_action!(planner)
    a = ordered_actions[ai]
    s_querry = s[s_idx]

    # do nonroot stuff
    depth = 0
    while true
        depth += 1
        @assert depth < 1 + length(s) "Loop has spiraled out of control!"

        if isterminalbelief(s[s_idx])
            s_querry = s[s_idx]
            break
        elseif length(sa_children[sa_idx]) < k_o * Nha[sa_idx] ^ alpha_o
            s_querry, r = @gen(:sp, :r)(mdp, s[s_idx], a, rng)
            s_idx = insert_state!(planner, s_querry; r, logits=zeros(Float32, length(ordered_actions)))
            break
        else
            s_idx = argmin(_bp_idx -> Nh[_bp_idx], sa_children[sa_idx])
            a, sa_idx = select_nonroot_action!(planner, s_idx)
        end
    end

    return s_querry, s_idx
end

function mcts_backward!(planner::GumbelSearch, value, logits)
    s_idx = planner.s_idx

    planner.tree.prior_value[s_idx]  = value
    planner.tree.prior_logits[s_idx] = logits
    planner.tree.prior_policy[s_idx] = softmax(logits)

    update_dq!(planner, value)

    if s_idx == 1
        mcts_backward_root!(planner)
    else
        mcts_backward_nonroot!(planner, s_idx)
    end

    return nothing
end

function mcts_backward_root!(planner::GumbelSearch)
    (; tree, live_actions, rng, m_acts_init) = planner

    root_logits = tree.prior_logits[1]

    # add gumbel noise to root logits for sampling and action selection
    for i in eachindex(root_logits)
        root_logits[i] += -log(-log(rand(rng, eltype(root_logits))))
    end

    # sample `m_acts_init` actions without replacement and insert them into the tree
    fill!(live_actions, false)
    for ai in partialsortperm(root_logits, 1:m_acts_init; rev=true)
        live_actions[ai] = true
        insert_action!(tree, 1, ai)
    end

    return nothing
end

function mcts_backward_nonroot!(planner::GumbelSearch, s_idx::Int)
    (; tree, mdp) = planner
    (; b_rewards, s_parent, sa_parent, Nh, Nha, Qha, prior_value) = tree

    gamma = discount(mdp)

    sa_idx = sa_parent[s_idx]
    value = prior_value[s_idx]

    while !iszero(sa_idx)
        value = b_rewards[s_idx] + gamma * value

        Nha[sa_idx] += 1
        Qha[sa_idx] += (value - Qha[sa_idx]) / Nha[sa_idx]

        update_dq!(planner, Qha[sa_idx])

        s_idx = s_parent[sa_idx]
        sa_idx = sa_parent[s_idx]

        Nh[s_idx] += 1
    end

    return nothing
end

function select_root_action!(planner::GumbelSearch)
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
            reduce_root_actions!(planner)
        end
    end

    ai, sa_idx = argmin(
        (ai, sa_idx) -> Nha[sa_idx],
        ((ai, sa_idx) for (ai, sa_idx) in s_children[1] if live_actions[ai])
    )

    return ai, sa_idx
end

function reduce_root_actions!(planner::GumbelSearch)
    (; tree, live_actions) = planner
    (; Qha, s_children) = tree

    root_logits = tree.prior_logits[1]

    sigma = get_sigma(planner, 1)

    for _ in 1:floor(Int, count(live_actions) / 2)
        ai, _ = argmin(
            (ai, sa_idx) -> root_logits[ai] + sigma * Qha[sa_idx],
            ((ai, sa_idx) for (ai, sa_idx) in s_children[1] if live_actions[ai])
        )
        live_actions[ai] = false
    end

    nothing
end

function select_nonroot_action!(planner::GumbelSearch, s_idx::Int)
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

function improved_policy(planner::GumbelSearch, s_idx::Int)
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

function get_sigma(planner::GumbelSearch, s_idx::Int; eps=1e-6, global_dq = true)
    (; tree, cscale, cvisit) = planner
    (; Nha, s_children, prior_value) = tree

    Nmax = maximum(Nha[sa_idx] for (_, sa_idx) in s_children[s_idx]; init = 0)

    if global_dq
        qmin, qmax = planner.q_extrema
    else
        qmin = minimum(Qha[sa_idx] for (_, sa_idx) in s_children[s_idx]; init =  Inf)
        qmax = maximum(Qha[sa_idx] for (_, sa_idx) in s_children[s_idx]; init = -Inf)
        qmin = min(qmin, prior_value[s_idx])
        qmax = max(qmax, prior_value[s_idx])
    end
    dq = max(qmax) - min(qmin)
    dq = dq < eps ? one(dq) : dq

    return cscale * (cvisit + Nmax) / dq
end

function update_dq!(planner::GumbelSearch, q)
    if q < planner.q_extrema[1]
        planner.q_extrema[1] = q
    elseif q > planner.q_extrema[2]
        planner.q_extrema[2] = q
    end
    return nothing
end
