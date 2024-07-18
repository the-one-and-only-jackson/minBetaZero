module BeliefMDPs

using POMDPs, POMDPTools
using Statistics, StatsBase, Distributions, Random
using StaticArrays, SparseArrays

export ExactBeliefMDP

struct ExactBeliefMDP{S, A, P, V, T} <: MDP{S,A}
    pomdp::P
    ordered_states::V
    b0::S
    transition_matrices::Dict{A, T}

    function ExactBeliefMDP(pomdp::P) where {S, A, O, P <: POMDP{S,A,O}}
        os = ordered_states(pomdp)

        etype = Float32

        b0 = if length(os) <= 100
            @MVector zeros(etype, length(os))
        else
            zeros(etype, length(os))
        end

        for (s, p) in weighted_iterator(initialstate(pomdp))
            sidx = stateindex(pomdp, s)
            b0[sidx] = p
        end

        T = transition_matrices(pomdp; sparse=true)
        for (k, v) in T
            T[k] = copy(v') # make each col correspond to the state index
        end

        new{typeof(b0), A, P, Vector{S}, valtype(T)}(pomdp, os, b0, T)
    end
end

POMDPs.actionindex(m::ExactBeliefMDP{B,A}, a::A) where {B,A} = actionindex(m.pomdp, a)
POMDPs.actions(m::ExactBeliefMDP) = actions(m.pomdp)

POMDPs.convert_a(::Type{V}, a, m::ExactBeliefMDP) where V<:AbstractArray = convert_a(V, a, m.pomdp)
POMDPs.convert_a(::Type{A}, vec::V, m::ExactBeliefMDP) where {A,V<:AbstractArray} = convert_a(A, vec, m.pomdp)

POMDPs.discount(m::ExactBeliefMDP) = discount(m.pomdp)

POMDPs.initialstate(m::ExactBeliefMDP) = Deterministic(m.b0)

function POMDPs.gen(m::ExactBeliefMDP{B,A}, b::B, a::A, rng::AbstractRNG) where {B, A}
    (; pomdp, ordered_states) = m

    T = m.transition_matrices[a]

    o = sample_obs(rng, pomdp, b, a, ordered_states, T)

    state_itr = Iterators.filter(
        (_, _, p_s)::Tuple -> !iszero(p_s),
        zip(eachindex(ordered_states), ordered_states, b)
    )

    function transition_itr(si)
        sparse_indicies = nzrange(T, si)
        spi     = @view rowvals(T)[sparse_indicies]
        sp      = @view ordered_states[spi]
        p_sp_s  = @view nonzeros(T)[sparse_indicies] # p(sp | s, a)
        return zip(spi, sp, p_sp_s)
    end

    bp = fill!(similar(b), 0)
    r  = 0.0

    for (si, s, p_s) in state_itr, (spi, sp, p_sp_s) in transition_itr(si)
        bp[spi] += p_sp_s * p_s * obs_weight(pomdp, s, a, sp, o)
        r       += p_sp_s * p_s * reward(pomdp, s, a, sp, o)
    end

    bp ./= sum(bp)

    return (; sp = bp, r)
end

function sample_obs(rng, pomdp, b, a, state_space, T)
    si   = sampleindex(rng, b)
    spi  = sampleindex(rng, nonzeros(T), nzrange(T, si))
    sp   = state_space[rowvals(T)[spi]]
    return rand(rng, observation(pomdp, a, sp))
end

function sampleindex(rng::AbstractRNG, b::AbstractVector{T}, indicies = eachindex(b)) where T
    @assert length(indicies) > 0

    b_sum = sum(@view b[indicies])

    @assert b_sum > eps(T) "indicies = $indicies, b = $b"

    u = rand(rng, T) * b_sum
    i = 0

    for outer i in Iterators.filter(i -> b[i] > eps(T), indicies)
        @inbounds u -= b[i]
        u <= 0 && break
    end

    return i
end


function POMDPs.isterminal(m::ExactBeliefMDP{B}, b::B) where B
    all(zip(b, m.ordered_states)) do (p, s)::Tuple
        p < eps(eltype(B)) || isterminal(m.pomdp, s)
    end
end

end
