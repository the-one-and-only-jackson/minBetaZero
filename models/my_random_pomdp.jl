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

    return MyRandomPOMDP{num_states,num_actions,num_observations}(S, A, Z, T, R, O)
end

POMDPs.states(m::MyRandomPOMDP) = m.states
POMDPs.stateindex(m::MyRandomPOMDP, s::SVector{1,Int}) = s[1]
POMDPs.initialstate(m::MyRandomPOMDP) = Uniform(m.states)
POMDPs.actions(m::MyRandomPOMDP) = m.actions
POMDPs.actionindex(m::MyRandomPOMDP, a) = a
POMDPs.observations(m::MyRandomPOMDP) = m.observations
POMDPs.obsindex(m::MyRandomPOMDP, o) = o
POMDPs.discount(m::MyRandomPOMDP) = 0.97
POMDPs.reward(m::MyRandomPOMDP, s, a) = m.R[s[1], a]
POMDPs.transition(m::MyRandomPOMDP, s, a) = SparseCat(m.states, @view m.T[s[1], :, a])
POMDPs.observation(m::MyRandomPOMDP, a, sp) = SparseCat(m.observations, @view m.O[:, sp[1]])


end


#=
Code to make things work and check

include("./models/my_random_pomdp.jl")
using .RandomPOMDP
using POMDPs
using StaticArrays

d = MyRandomPOMDP()
s = SVector(11)
transition(d,s,3)
observation(d,:right,s)
b = initialstate(d)
reward(d,s,2)

using QMDP
solver = QMDPSolver(max_iterations=2000,belres=1e-3,verbose=true)
policy = solve(solver, d);
a = action(policy,b)

using SARSOP
solver = SARSOPSolver()
policy = solve(solver, d);
a = action(policy,b)

import POMDPTools:RolloutSimulator
using ProgressMeter
using Random
using Statistics
sim_rng = MersenneTwister(21)
n_episodes = 100
max_steps = 100
returns = Vector{Union{Float64,Missing}}(missing, n_episodes);
up = updater(policy);
# up = BootstrapFilter(d, 100, MersenneTwister(abs(rand(Int8))));
@showprogress for i in 1:n_episodes
    ro = RolloutSimulator(max_steps=max_steps, rng=sim_rng)
    returns[i] = simulate(ro, d, policy, up)
end
mean(returns)
std(returns)/sqrt(n_episodes)

=#
