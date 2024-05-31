module LightDark

using POMDPs
using Distributions: Uniform, Normal
using POMDPModelTools: ImplicitDistribution, Deterministic

export LightDarkPOMDP, LightDarkState

struct LightDarkState
    status::Int64
    y::Float64
end

@kwdef mutable struct LightDarkPOMDP <: POMDPs.POMDP{LightDarkState,Int,Float64}
    discount_factor::Float64 = 0.9
    correct_r::Float64 = 100.0
    incorrect_r::Float64 = -100.0
    step_size::Float64 = 1.0
    movement_cost::Float64 = 0.0
    max_y::Float64 = 100.0
    light_loc::Float64 = 10.0
    sigma::Function = y->abs(y - light_loc) + 1e-4
end

POMDPs.discount(p::LightDarkPOMDP) = p.discount_factor
POMDPs.isterminal(::LightDarkPOMDP, act::Int64) = act == 0
POMDPs.isterminal(::LightDarkPOMDP, s::LightDarkState) = s.status < 0
POMDPs.actions(::LightDarkPOMDP) = -1:1
POMDPs.actionindex(::LightDarkPOMDP, a) = a+2

function POMDPs.initialstate(::LightDarkPOMDP; isuniform::Bool=false)
    dist = isuniform ?  Uniform(-30, 30) : Normal(2, 3)
    ImplicitDistribution() do rng
        LightDarkState(0, rand(rng, dist))
    end
end
POMDPs.initialobs(m::LightDarkPOMDP, s) = observation(m, s)

POMDPs.observation(p::LightDarkPOMDP, sp::LightDarkState) = Normal(sp.y, p.sigma(sp.y))

function POMDPs.transition(p::LightDarkPOMDP, s::LightDarkState, a::Int)
    @assert a ∈ actions(p) "Action $a"
    status = (a == 0) ? -1 : s.status
    a = clamp(a, -1, 1)
    y = clamp(s.y + a*p.step_size, -p.max_y, p.max_y)
    return Deterministic(LightDarkState(status, y))
end

function POMDPs.reward(p::LightDarkPOMDP, s::LightDarkState, a::Int)
    s.status < 0 && return 0.0
    a == 0       && return abs(s.y) < 1 ? p.correct_r : p.incorrect_r
                    return -p.movement_cost
end

POMDPs.convert_s(::Type{A}, s::LightDarkState, p::LightDarkPOMDP) where A<:AbstractArray = eltype(A)[s.status, s.y]
POMDPs.convert_s(::Type{LightDarkState}, s::A, p::LightDarkPOMDP) where A<:AbstractArray = LightDarkState(Int64(s[1]), Float64(s[2]))

end # module LightDark
