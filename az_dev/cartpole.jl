module CartPole

using POMDPs, POMDPTools, StaticArrays, Distributions

export CartPoleMDP

@kwdef struct CartPoleMDP <: MDP{SVector{4, Float64}, Int}
    g               :: Float64 = 9.8
    mass_cart       :: Float64 = 1.0
    mass_pole       :: Float64 = 0.5
    length          :: Float64 = 0.5
    force_mag       :: Float64 = 10.
    dt              :: Float64 = 0.02
    theta_max       :: Float64 = 12*pi/180
    x_max           :: Float64 = 2.4
end

POMDPs.discount(::CartPoleMDP) = 0.99

POMDPs.actions(::CartPoleMDP) = SA[1, 2]
POMDPs.actionindex(::CartPoleMDP, a::Int) = a

function POMDPs.isterminal(env::CartPoleMDP, s)
    abs(s[1]) > env.x_max || abs(s[3]) > env.theta_max
end

function POMDPs.initialstate(::CartPoleMDP)
    ImplicitDistribution() do rng
        (1 // 10) * (@SArray(rand(rng, 4)) .- (1//2))
    end
end

POMDPs.reward(env::CartPoleMDP, s, a) = isterminal(env, s) ? 0.0 : 1.0

function POMDPs.transition(env::CartPoleMDP, state::SVector{4, Float64}, action::Int)
    (; g, mass_cart, mass_pole, length, force_mag, dt) = env

    @assert action ∈ POMDPs.actions(env) "Action out of bounds"

    x, x_dot, theta, theta_dot = state

    force = action == 1 ? force_mag : -force_mag

    costheta, sintheta = cos(theta), sin(theta)

    temp = (force + (mass_pole * length) * theta_dot^2 * sintheta) / (mass_cart + mass_pole)
    thetaacc_num = g * sintheta - costheta * temp
    thetaacc_den = length * (4//3 - mass_pole * costheta^2 / (mass_cart + mass_pole))
    thetaacc = thetaacc_num / thetaacc_den
    xacc = temp - mass_pole * length * thetaacc * costheta / (mass_cart + mass_pole)

    newstate = state .+ dt * SA[x_dot, xacc, theta_dot, thetaacc]

    return Deterministic(newstate)
end

end
