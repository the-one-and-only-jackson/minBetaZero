mutable struct History{S, P <: AbstractVector, V <: AbstractFloat}
    state           :: Vector{S}
    reward          :: Vector{V}
    value_target    :: Vector{V}
    policy_target   :: Vector{P}
    steps           :: Int
    episode_reward  :: V
    discount        :: V
    current_state   :: S

    function History{S, P, V}(state, reward, value_target, policy_target, steps, episode_reward, discount, current_state) where {S, P, V}
        return new{S,P,V}(state, reward, value_target, policy_target, steps, episode_reward, discount, current_state)
    end

    function History{S, P, V}(current_state::S, discount = 1) where {S, P, V}
        return new{S,P,V}(S[], V[], V[], P[], 0, zero(V), V(discount), current_state)
    end
    History{S, P}(current_state::S, discount = 1) where {S, P} = History{S, P, Float32}(current_state, discount)
    History{S}(current_state::S, discount = 1) where S = History{S, Vector{Float32}}(current_state, discount)
end
History(m::MDP{S}, current_state::S) where S = History{S}(current_state, discount(m))

function copy_and_reset!(h::H, newstate::S) where {S, P, V, H <: History{S, P, V}}
    new_h = H(
        h.state,
        h.reward,
        h.value_target,
        h.policy_target,
        h.steps,
        h.episode_reward,
        h.discount,
        h.current_state
    )
    h.state = S[]
    h.reward = V[]
    h.value_target = V[]
    h.policy_target = P[]
    h.steps = 0
    h.episode_reward = 0
    h.current_state = newstate
    return new_h
end

function reset!(h::History{S}, newstate::S) where S
    empty!(h.state)
    empty!(h.reward)
    empty!(h.value_target)
    empty!(h.policy_target)
    h.steps = 0
    h.episode_reward = 0
    h.current_state = newstate
    return nothing
end

function Base.push!(h::History{S}; sp::S, reward, policy_target, value_target = nothing) where S
    push!(h.state, h.current_state)
    push!(h.reward, reward)
    push!(h.policy_target, policy_target)
    isnothing(value_target) || push!(h.value_target, value_target)

    h.episode_reward += reward * h.discount ^ h.steps
    h.steps += 1
    h.current_state = sp

    return h
end

function calculate_value_target(h::History, v_end = 0)
    isempty(h.value_target) || return nothing
    v_end = typeof(h.reward)(v_end)
    resize!(h.value_target, h.steps)
    for i in h.steps:-1:1
        h.value_target[i] = v_end = h.reward[i] + h.discount * v_end
    end
    return nothing
end
