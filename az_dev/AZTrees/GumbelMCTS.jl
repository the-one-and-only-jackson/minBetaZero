struct GumbelSearch{Tree, A, M, RNG}
    mdp             :: M
    tree            :: Tree
    tree_querries   :: Int
    ordered_actions :: Vector{A}
    m_acts_init     :: Int
    live_actions    :: Vector{Bool}
    target_N        :: SeqHalf
    k_o             :: Float64
    alpha_o         :: Float64
    cscale          :: Float64
    cvisit          :: Float64
    rng             :: RNG
    policy_scratch  :: Vector{Float32}

    function GumbelSearch(mdp :: M;
            tree_querries :: Real = length(actions(mdp)),
            m_acts_init   :: Real = length(actions(mdp)),
            k_o           :: Real = 1,
            alpha_o       :: Real = 0,
            cscale        :: Real = 0.1,
            cvisit        :: Real = 50,
            rng           :: RNG  = Random.default_rng()
        ) where {S, A, M <: MDP{S, A}, RNG <: AbstractRNG}

        na = length(actions(mdp))

        tree_querries   = floor(Int, tree_querries) # root querry addition
        ordered_actions = POMDPTools.ordered_actions(mdp)
        m_acts_init     = floor(Int, m_acts_init)
        live_actions    = [false for _ in 1:na]
        target_N        = SeqHalf(; n=tree_querries, m=m_acts_init)

        k_o     = Float64(k_o)
        alpha_o = Float64(alpha_o)
        cscale  = Float64(cscale)
        cvisit  = Float64(cvisit)

        tree = GuidedTree{S, Int32, Float32}(tree_querries + 1, na, ceil(Int, k_o))

        policy_scratch = zeros(Float32, na)

        return new{GuidedTree{S, Int32, Float32}, A, M, RNG}(
            mdp,
            tree,
            tree_querries,
            ordered_actions,
            m_acts_init,
            live_actions,
            target_N,
            k_o,
            alpha_o,
            cscale,
            cvisit,
            rng,
            policy_scratch
        )
    end
end

function root_info(planner::GumbelSearch)
    a = select_best_action(planner)
    policy_target, v_mix = get_improved_policy(planner, 1)
    return a, (; planner.tree, policy_target = copy(policy_target), value_target = v_mix)
end

function live_root_actions(planner::GumbelSearch)
    Iterators.filter(
        (ai, sa_idx)::Tuple -> planner.live_actions[ai],
        s_children(planner.tree, 1)
    )
end

function select_best_action(planner::GumbelSearch)
    (; tree, ordered_actions) = planner
    (; Qha, prior_logits) = tree

    sigma = get_sigma(planner, 1)

    ai_opt, _ = argmax(
        (ai, sa_idx)::Tuple -> prior_logits[ai, 1] + sigma * Qha[sa_idx],
        live_root_actions(planner)
    )

    a = ordered_actions[ai_opt]

    return a
end

Base.isdone(planner::GumbelSearch) = planner.tree.Nh[1] >= planner.tree_querries

function insert_root!(planner::GumbelSearch, s_root)
    (; tree, mdp, target_N) = planner

    @assert !isterminal(mdp, s_root) "Root state is terminal! s = $s_root"

    reset!(target_N)
    reset!(tree)
    insert_state!(tree, s_root)

    return nothing
end

function mcts_forward!(planner::GumbelSearch)
    new_root = iszero(n_s_children(planner.tree, 1))
    if new_root
        return planner.tree.state[1]
    else
        s_idx = mcts_forward_nonroot!(planner)
        return planner.tree.state[s_idx]
    end
end

function mcts_forward_nonroot!(planner::GumbelSearch)
    (; tree, k_o, alpha_o, mdp, rng) = planner
    (; Nh, Nha, state) = tree

    s_idx = 1
    exit_flag = false

    while !exit_flag
        a, sa_idx = select_action!(planner, s_idx)

        if n_sa_children(tree, sa_idx) < k_o * Nha[sa_idx] ^ alpha_o
            s_querry, r = @gen(:sp, :r)(mdp, state[s_idx], a, rng)
            s_idx = insert_state!(tree, s_querry, sa_idx, r)
            exit_flag = true
        else
            s_idx = argmin(i -> Nh[i], sa_children(tree, sa_idx))
            exit_flag = isterminal(mdp, state[s_idx])
        end

        push_stack!(tree, sa_idx, s_idx)
    end

    return s_idx
end

function mcts_backward!(planner::GumbelSearch, value, logits)
    if iszero(planner.tree.sa_counter)
        mcts_backward_root!(planner, value, logits)
    else
        mcts_backward_nonroot!(planner, value, logits)
    end
end

function mcts_backward_root!(planner::GumbelSearch, value, logits)
    (; tree, live_actions, rng, m_acts_init) = planner
    (; prior_logits) = tree

    update_prior!(tree, 1, logits, value)

    # add gumbel noise to root logits for sampling and action selection
    for i in 1:size(prior_logits, 1)
        u = rand(rng, eltype(prior_logits))
        prior_logits[i, 1] += -log(-log(u))
    end

    # sample `m_acts_init` actions without replacement and insert them into the tree
    fill!(live_actions, false)

    for _ in 1:m_acts_init
        ai = argmax(
            ai -> prior_logits[ai, 1],
            Iterators.filter(ai -> !live_actions[ai], 1:length(live_actions))
        )
        live_actions[ai] = true
        insert_action!(tree, 1, ai)
    end

    return nothing
end

function mcts_backward_nonroot!(planner::GumbelSearch, value::Real, logits)
    (; tree, mdp) = planner
    (; reward, Nh, Nha, Qha) = tree

    sa_idx, s_idx = pop_stack!(tree)

    update_prior!(tree, s_idx, logits, value)

    Nha[sa_idx] += 1
    Qha[sa_idx] += (value - Qha[sa_idx]) / Nha[sa_idx]
    update_dq!(tree, Qha[sa_idx])

    value = eltype(Qha)(value)
    gamma = eltype(Qha)(discount(mdp))

    while !stack_empty(tree)
        sa_idx, s_idx = pop_stack!(tree)

        value = reward[s_idx] + gamma * value

        Nh[s_idx] += 1
        Nha[sa_idx] += 1
        Qha[sa_idx] += (value - Qha[sa_idx]) / Nha[sa_idx]
        update_dq!(tree, Qha[sa_idx])
    end

    Nh[1] += 1

    return nothing
end

function select_action!(planner::GumbelSearch, s_idx::Integer)
    if isone(s_idx)
        select_root_action!(planner)
    else
        select_nonroot_action!(planner, s_idx)
    end
end

function select_root_action!(planner::GumbelSearch)
    (; tree, live_actions, target_N, ordered_actions) = planner
    (; Nha) = tree

    halving_flag = !any(
        (ai, sa_idx)::Tuple -> Nha[sa_idx] < target_N.N,
        live_root_actions(planner)
    )

    if halving_flag
        next!(target_N)
        if count(live_actions) > 2
            reduce_root_actions!(planner)
        end
    end

    ai, sa_idx = argmin(
        (ai, sa_idx)::Tuple -> Nha[sa_idx],
        live_root_actions(planner)
    )

    a = ordered_actions[ai]

    return a, sa_idx
end

function reduce_root_actions!(planner::GumbelSearch)
    (; tree, live_actions) = planner
    (; Qha, prior_logits) = tree

    sigma = get_sigma(planner, 1)

    for _ in 1:floor(Int, count(live_actions) / 2)
        ai, _ = argmin(
            (ai, sa_idx)::Tuple -> prior_logits[ai, 1] + sigma * Qha[sa_idx],
            live_root_actions(planner)
        )
        live_actions[ai] = false
    end

    nothing
end

function select_nonroot_action!(planner::GumbelSearch, s_idx::Integer)
    (; tree, ordered_actions) = planner
    (; Nh, Nha) = tree

    pi_completed, _ = get_improved_policy(planner, s_idx)

    max_target = pi_completed
    for (ai, sa_idx) in s_children(tree, s_idx)
        max_target[ai] -= Nha[sa_idx] / (1 + Nh[s_idx])
    end

    ai     = argmax(max_target)
    a      = ordered_actions[ai]
    sa_idx = insert_action!(tree, s_idx, ai)

    return a, sa_idx
end

function get_improved_policy(planner::GumbelSearch, s_idx::Integer)
    (; tree, policy_scratch) = planner
    (; Qha, Nh, prior_logits, prior_value) = tree
    temp = policy_scratch # Use planner.policy_scratch as a preallocated array

    policy = softmax!(temp, @view prior_logits[:, s_idx])

    sum_pi   = zero(eltype(policy))
    sum_pi_q = zero(promote_type(eltype(Qha), eltype(policy)))
    for (ai, sa_idx) in s_children(tree, s_idx)
        sum_pi   += policy[ai]
        sum_pi_q += policy[ai] * Qha[sa_idx]
    end
    visited_value = sum_pi_q / sum_pi

    w1    =         1 // (1 + Nh[s_idx])
    w2    = Nh[s_idx] // (1 + Nh[s_idx])
    v_mix = w1 * prior_value[s_idx] + w2 * visited_value

    sigma = get_sigma(planner, s_idx)

    improved_logits = temp .= @view prior_logits[:, s_idx]
    improved_logits .+= sigma * v_mix # constant offset doesn't change softmax
    for (ai, sa_idx) in s_children(tree, s_idx)
        # transform is monotonically increasing, so okay to use (policy improvement)
        advantage = Qha[sa_idx] - v_mix
        transformed_advantage = sigma * advantage
        improved_logits[ai] += transformed_advantage
    end

    improved_policy = softmax!(improved_logits)

    return improved_policy, v_mix
end

softmax!(x) = softmax!(x, x)
function softmax!(y, x)
    copyto!(y, x)
    y .-= maximum(y)
    y .= exp.(y)
    y ./= sum(y)
end

function get_sigma(planner::GumbelSearch, s_idx::Integer; eps=1e-6, global_dq=true)
    (; tree, cscale, cvisit) = planner

    dq = get_dq(planner.tree, s_idx; eps, global_dq)

    Nmax = maximum(
        (_, sa_idx)::Tuple -> tree.Nha[sa_idx],
        s_children(tree, s_idx);
        init = zero(eltype(tree.Nha))
    )

    sigma = cscale * (cvisit + Nmax) / dq

    return sigma
end
