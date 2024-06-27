@kwdef mutable struct History{S, T <: Real}
    state           :: Vector{S}         = S[]
    reward          :: Vector{T}         = T[]
    value_target    :: Vector{T}         = T[]
    policy_target   :: Vector{Vector{T}} = Vector{T}[]
    steps           :: Int               = 0
    episode_reward  :: T                 = zero(T)
    discount        :: T                 = one(T)
    current_state   :: S
end

function History(m::MDP{S}, current_state::S) where S
    History{S, Float32}(; current_state, discount = Float32(discount(m)))
end

function copy_and_reset!(h::H, newstate::S) where {S, T, H <: History{S, T}}
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
    h.reward = T[]
    h.value_target = T[]
    h.policy_target = Vector{T}[]
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
