struct ExactBeliefMDP{S, A, P, V, T} <: MDP{S,A}
    pomdp::P
    ordered_states::V
    b0::S
    transition_matrices::Dict{A, T}
    function ExactBeliefMDP(pomdp::P) where {S, A, O, P <: POMDP{S,A,O}}
        os = ordered_states(pomdp)

        b0 = zeros(Float32, length(os))
        for (s, p) in weighted_iterator(initialstate(pomdp))
            sidx = stateindex(pomdp, s)
            b0[sidx] = p
        end

        T = transition_matrices(pomdp; sparse=true)
        TT = valtype(T)
        for (k,v) in T
            T[k] = copy(v') # make each col correspond to the state index
        end

        new{Vector{Float32}, A, P, Vector{S}, TT}(pomdp, os, b0, T)
    end
end

POMDPs.actionindex(m::ExactBeliefMDP{B,A}, a::A) where {B,A} = actionindex(m.pomdp, a)
POMDPs.actions(m::ExactBeliefMDP) = actions(m.pomdp)

POMDPs.convert_a(::Type{V}, a, m::ExactBeliefMDP) where V<:AbstractArray = convert_a(V, a, m.pomdp)
POMDPs.convert_a(::Type{A}, vec::V, m::ExactBeliefMDP) where {A,V<:AbstractArray} = convert_a(A, vec, m.pomdp)

POMDPs.discount(m::ExactBeliefMDP) = discount(m.pomdp)

POMDPs.initialstate(m::ExactBeliefMDP) = Deterministic(m.b0)

function POMDPs.gen(m::ExactBeliefMDP{B,A}, b::B, a::A, rng::AbstractRNG) where {B, A}
    pomdp = m.pomdp
    state_space = m.ordered_states

    T = m.transition_matrices[a]
    Trows = rowvals(T)
    Tvals = nonzeros(T)

    o = sample_obs(rng, pomdp, b, a, state_space, T)

    bp = similar(b)
    fill!(bp, 0)

    r  = 0.0

    for (si, (s, b_s)) in enumerate(zip(state_space, b))
        iszero(b_s) && continue
        for i in nzrange(T, si)
            spi = Trows[i]
            sp = state_space[spi]
            op = obs_weight(pomdp, s, a, sp, o)

            tp = Tvals[i]
            bp[spi] += op * tp * b_s
            r += tp * b_s * reward(pomdp, s, a, sp)
        end
    end

    bp ./= sum(bp)

    return (; sp = bp, r)
end

function sample_obs(rng, pomdp, b, a, state_space, T)
    si = sampleindex(rng, b)

    Trows = rowvals(T)
    Tvals = nonzeros(T)
    idxs = nzrange(T, si)

    spi = sampleindex(rng, Tvals, first(idxs), last(idxs))
    sp = state_space[Trows[spi]]

    o = rand(rng, observation(pomdp, a, sp))

    return o
end

sampleindex(rng, b) = sampleindex(rng, b, 1, length(b))
function sampleindex(rng::AbstractRNG, b::AbstractVector{T}, first::Int, last::Int) where T
    u = rand(rng, T) * sum(@view b[first:last])
    i = first - 1
    while i < last && u > zero(T)
        i += 1
        @inbounds u -= b[i]
    end
    return i
end


function POMDPs.isterminal(m::ExactBeliefMDP{B}, b::B) where B
    for (s, p) in zip(m.ordered_states, b)
        if p > eps() && !isterminal(m.pomdp, s)
            return false
        end
    end
    return true
end
