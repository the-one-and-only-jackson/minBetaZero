module RandomPOMDP

using POMDPs, Random, StaticArrays, POMDPTools

export MyRandomPOMDP


struct MyRandomPOMDP{M,N,P} <: POMDP{SVector{1,Int}, Int, Int}
    states::Array{SVector{1,Int},1}
    actions::SVector{N,Int}
    observations::SVector{P,Int}
    T::Array{Float64, 3}
    R::Array{Float64, 2}
    O::Array{Float64, 2}
    initial_states::Array{SVector{1,Int},1}
    term_idx::Int
end

function MyRandomPOMDP(;
    num_states = 100,
    num_actions = 4,
    num_observations = 10,
    rng::AbstractRNG = Random.MersenneTwister(20)
    )

    S = [SVector(i) for i in 1:num_states]
    A = collect(1:num_actions)
    Z = collect(1:num_observations)

    #Define the Transition function and make it a valid probability distribution
    T = rand(rng, num_states, num_states, num_actions)
    for i in 1:num_actions
        T[:, :, i] ./= sum(T[:, :, i], dims=2)
    end

    #Define the Reward function
    R = rand(rng, num_states, num_actions)

    #Define the Observation function and make it a valid probability distribution
    #Sum of the Probability of observation should add upto 1 for all states.
    O = rand(rng, num_observations, num_states)
    for i in 1:num_states
        O[:,i] ./= sum(O[:,i])
    end

    # add terminal behaviour
    term_idx = rand(rng, 1:num_states)

    for i in 1:num_actions
        T[term_idx, :, i] .= 0
        T[term_idx, term_idx, i] = 1
    end

    R[term_idx, :] .= 0

    O[end, :] .= 0
    O[:, term_idx] .= 0
    O[end, term_idx] = 1
    O ./= sum(O; dims=2)

    initial_states = [SVector(i) for i in 1:num_states if i != term_idx]

    return MyRandomPOMDP{num_states,num_actions,num_observations}(S, A, Z, T, R, O, initial_states, term_idx)
end

POMDPs.states(m::MyRandomPOMDP) = m.states
POMDPs.stateindex(m::MyRandomPOMDP, s::SVector{1,Int}) = s[1]
POMDPs.initialstate(m::MyRandomPOMDP) = UnsafeUniform(m.initial_states)
POMDPs.actions(m::MyRandomPOMDP) = m.actions
POMDPs.actionindex(m::MyRandomPOMDP, a) = a
POMDPs.observations(m::MyRandomPOMDP) = m.observations
POMDPs.obsindex(m::MyRandomPOMDP, o) = o
POMDPs.discount(m::MyRandomPOMDP) = 0.97
POMDPs.reward(m::MyRandomPOMDP, s, a) = m.R[s[1], a]
POMDPs.transition(m::MyRandomPOMDP, s, a) = SparseCat(m.states, @view m.T[s[1], :, a])
POMDPs.observation(m::MyRandomPOMDP, a, sp) = SparseCat(m.observations, @view m.O[:, sp[1]])
POMDPs.isterminal(m::MyRandomPOMDP, s) = stateindex(m,s) == m.term_idx

end
