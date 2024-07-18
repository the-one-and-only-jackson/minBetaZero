@kwdef mutable struct MDPHistory{S, T <: Real}
    state           :: Vector{S}         = S[]
    reward          :: Vector{T}         = T[]
    value_target    :: Vector{T}         = T[]
    policy_target   :: Vector{Vector{T}} = Vector{T}[]
    steps           :: Int               = 0
    episode_reward  :: T                 = zero(T)
    trajectory_done :: Bool              = false
end

MDPHistory(::MDP{S}) where S = MDPHistory{S, Float32}()

Base.length(h::MDPHistory) = length(h.state)

function copy_and_reset!(h::MDPHistory{S,T}) where {S, T}
    new_h = MDPHistory(
        h.state,
        h.reward,
        h.value_target,
        h.policy_target,
        h.steps,
        h.episode_reward,
        h.trajectory_done
    )
    h.state = S[]
    h.reward = T[]
    h.value_target = T[]
    h.policy_target = Vector{T}[]
    if h.trajectory_done
        h.steps = 0
        h.episode_reward = 0
        h.trajectory_done = false
    end
    return new_h
end

function calculate_value_target(h::MDPHistory{S,T}, v_end = 0; discount = 1) where {S,T}
    (; value_target, reward) = h

    isempty(value_target) || return nothing

    v_end    = T(v_end)
    discount = T(discount)

    resize!(value_target, length(h))
    for i in Iterators.reverse(eachindex(value_target))
        value_target[i] = v_end = reward[i] + discount * v_end
    end

    return nothing
end

struct MDPAgent{M <: MDP, MCTS <: GumbelSearch, RNG <: AbstractRNG, H <: MDPHistory}
    mdp             :: M
    mcts            :: MCTS
    rng             :: RNG
    history         :: H
    max_steps       :: Int
    segment_length  :: Int
end

function MDPAgent(mdp::MDP, params::AlphaZeroParams)
    (; rng, max_steps, segment_length, tree_queries, k_o, cscale, cvisit, m_acts_init) = params

    mctsargs = (; tree_queries, k_o, cscale, cvisit, m_acts_init, rng)

    mcts    = GumbelSearch(mdp; mctsargs...)
    history = MDPHistory(mdp)
    state   = rand(rng, initialstate(mdp))
    push!(history.state, state)

    return MDPAgent(mdp, mcts, rng, history, Int(max_steps), Int(segment_length))
end

function initialize_agent!(agent::MDPAgent)
    (; mcts, history) = agent
    s = history.state[end]
    insert_root!(mcts, s)
    s_querry = mcts_forward!(mcts)
    return s_querry
end

function step_agent!(agent::MDPAgent, history_channel::Channel)
    (; mdp, mcts, rng, history, max_steps, segment_length) = agent

    s = history.state[end]
    a, a_info = root_info(mcts)
    sp, r = @gen(:sp, :r)(mdp, s, a, rng)

    push!(history.reward, r)
    push!(history.policy_target, a_info.policy_target)
    # push!(history.value_target, a_info.value_target)

    history.episode_reward += r * discount(mdp) ^ history.steps
    history.steps += 1

    terminated = isterminal(mdp, sp)
    truncated  = history.steps >= max_steps

    if terminated || truncated
        sp = rand(rng, initialstate(mdp))
        history.trajectory_done = true
    end

    if terminated || truncated || length(history) == segment_length
        v_end = terminated ? zero(a_info.next_value) : a_info.next_value
        calculate_value_target(history, v_end; discount = discount(mdp))

        history_copy = copy_and_reset!(history)
        put!(history_channel, history_copy)
    end

    push!(history.state, sp)
    insert_root!(mcts, sp)

    return nothing
end
