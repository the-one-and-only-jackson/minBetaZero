module OddEven

using POMDPs, POMDPTools
import Distributions: Normal, cdf

export OddEvenPOMDP

struct OddEvenPOMDP <: POMDP{Int, Int, Int}
    num_states::Int
    rlow::Float64
    rhigh::Float64
    discount::Float64
    O::Matrix{Float64}
end

function OddEvenPOMDP(; num_states = 100, rlow = -10, rhigh = 100, discount = 0.95)
    O = stack(1:num_states) do mu
        sigma = 5 + max(mu, num_states - mu) / 3
        discretize_gaussian(mu, sigma, num_states)
    end

    return OddEvenPOMDP(Int(num_states), Float64.((rlow, rhigh, discount))..., O)
end

POMDPs.states(m::OddEvenPOMDP) = 1:m.num_states
POMDPs.actions(m::OddEvenPOMDP) = 1:m.num_states
POMDPs.observations(m::OddEvenPOMDP) = 1:m.num_states

POMDPs.stateindex(::OddEvenPOMDP, s::Integer) = s
POMDPs.actionindex(::OddEvenPOMDP, a::Integer) = a
POMDPs.obsindex(::OddEvenPOMDP, o::Integer) = o


POMDPs.initialstate(m::OddEvenPOMDP) = UnsafeUniform(1:m.num_states)
POMDPs.transition(::OddEvenPOMDP, s::Integer, ::Integer) = Deterministic(s)
POMDPs.observation(m::OddEvenPOMDP, sp::Integer) = SparseCat(1:m.num_states, @view m.O[:, sp])

POMDPs.reward(m::OddEvenPOMDP, s::Integer, a::Integer) = s == a ? m.rhigh : m.rlow
POMDPs.discount(m::OddEvenPOMDP) = m.discount
POMDPs.isterminal(::OddEvenPOMDP, ::Integer) = false


function discretize_gaussian(μ::Integer, σ, num_bins)
    @assert 0 <= μ <= num_bins

    bin_edges = range(0, stop=num_bins, length=num_bins+1)
    cdf_values = cdf.(Normal(μ, σ), bin_edges)
    bin_probs = diff(cdf_values)

    init = iseven(μ) ? 1 : 2
    bin_probs[init:2:end] .= 0

    bin_probs /= sum(bin_probs)

    return bin_probs
end

end
