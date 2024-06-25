module LaserTag

using POMDPs, Random, StaticArrays, POMDPTools

export LTState, LaserTagPOMDP

struct LTState
    robot  :: SVector{2, Int}
    target :: SVector{2, Int}
end

Base.convert(::Type{SVector{4, Int}}    , s::LTState) = SA[s.robot..., s.target...]
Base.convert(::Type{AbstractVector{Int}}, s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractVector}     , s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractArray}      , s::LTState) = convert(SVector{4, Int}, s)

struct LaserTagPOMDP <: POMDP{LTState, Symbol, SVector{5,Int}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    initial_states::Vector{LTState}
    actions::Vector{Symbol}
    observations::Vector{SVector{5,Int}}
    obsindices::Array{Int, 5}
    obs_prob::Float64
end

function LaserTagPOMDP(;
    size = (10, 7),
    n_obstacles::Int = 9,
    rng::AbstractRNG = Random.MersenneTwister(20),
    obs_prob::Float64 = 0.1, # probability of recieving correct observation
    use_measure::Bool = true
    )

    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end

    condition = true
    local robot_init::SVector{2, Int}
    while condition
        robot_init = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        condition = robot_init in obstacles
    end

    initial_states = [
        LTState(SVector(xr, yr), SVector(xt, yt))
        for xr in 1:size[1]
        for yr in 1:size[2]
        for xt in 1:size[1]
        for yt in 1:size[2]
        if !in(SVector(xr, yr), obstacles) && !in(SVector(xt, yt), obstacles) && !(xt == xr && yt == yr)
    ]

    actions = [:left, :right, :up, :down, :left_up, :right_up, :left_down, :right_down]
    use_measure && push!(actions, :measure)

    observations = [
        SVector(same_grid, left, right, up, down)
        for same_grid in 0:1
        for left      in 0:size[1]-1
        for right     in 0:size[1]-1-left
        for up        in 0:size[2]-1
        for down      in 0:size[2]-1-up
    ]

    obsindices = zeros(Int, 2, size[1], size[1], size[2], size[2])
    for (ind, o) in enumerate(observations)
        obsindices[(o.+1)...] = ind
    end

    LaserTagPOMDP(SVector(size), obstacles, blocked, initial_states, actions, observations, obsindices, obs_prob)
end

const actiondir = Dict(
    :left=>SVector(-1,0), :right=>SVector(1,0), :up=>SVector(0, 1), :down=>SVector(0,-1),
    :measure=>SVector(0,0), :left_up=>SVector(-1,1), :right_up=>SVector(1,1),
    :left_down=>SVector(-1, -1), :right_down=>SVector(1,-1)
)

function POMDPs.states(m::LaserTagPOMDP)
    vec(collect(
        LTState(SVector(c[1], c[2]), SVector(c[3], c[4]))
        for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2])
    ))
end

POMDPs.initialstate(m::LaserTagPOMDP) = UnsafeUniform(m.initial_states)
POMDPs.actions(m::LaserTagPOMDP) = m.actions
POMDPs.observations(m::LaserTagPOMDP) = m.observations
POMDPs.discount(m::LaserTagPOMDP) = 0.99
POMDPs.actionindex(m::LaserTagPOMDP, a) = findfirst(==(a), m.actions)
POMDPs.obsindex(m::LaserTagPOMDP, o) = m.obsindices[(o .+ 1)...]
POMDPs.isterminal(::LaserTagPOMDP, s) = s.robot == s.target

function POMDPs.stateindex(m::LaserTagPOMDP, s::LTState)
    idxs = (1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2])
    LinearIndices(idxs)[s.robot..., s.target...]
end

function POMDPs.reward(m::LaserTagPOMDP, s::LTState, a::Symbol, sp::LTState)
    if isterminal(m, s)
        return 0.0
    elseif isterminal(m, sp)
        return 100.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end

function POMDPs.transition(m::LaserTagPOMDP, s::LTState, a::Symbol)
    if isterminal(m, s)
        return Deterministic(s)
    end

    newrobot = bounce(m, s.robot, actiondir[a])

    if sum(abs, newrobot - s.target) <= 2
        return move_target_away(m, s, newrobot)
    else
        return move_target_random(m, s, newrobot)
    end
end

function move_target_away(m::LaserTagPOMDP, s::LTState, newrobot)
    away = sign.(s.target - s.robot)
    if sum(abs, away) == 2 # diagonal
        away -= SVector(0, away[2]) # preference to move in x direction
    end
    newtarget = bounce(m, s.target, away)
    return Deterministic(LTState(newrobot, newtarget))
end

function move_target_random(m::LaserTagPOMDP, s::LTState, newrobot)
    nextstate = [LTState(newrobot, s.target)]
    nextstateprobs = Float64[0.0]

    for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
        newtarget = bounce(m, s.target, change)
        if newtarget == s.target
            nextstateprobs[1] += 0.25
        else
            push!(nextstate, LTState(newrobot, newtarget))
            push!(nextstateprobs, 0.25)
        end
    end

    return SparseCat(nextstate, nextstateprobs)
end

function bounce(m::LaserTagPOMDP, pos::SVector{2, Int}, change::SVector{2, Int})
    newpos = clamp.(pos + change, SVector(1,1), m.size)
    isblocked = m.blocked[newpos...]
    return isblocked ? pos : newpos
end

function POMDPs.observation(m::LaserTagPOMDP, a::Symbol, sp::LTState)
    left, down = sp.robot .- 1
    right, up = m.size - sp.robot

    ranges = SVector(left, right, up, down)
    for obstacle in m.obstacles
        ranges = laserbounce(ranges, sp.robot, obstacle)
    end
    ranges = laserbounce(ranges, sp.robot, sp.target)

    done = isterminal(m, sp)
    os = (SVector{5,Int}(done, ranges...), SVector{5,Int}(done, 0, 0, 0, 0))

    probs = a == :measure || all(iszero, ranges) ? (1.0, 0.0) : (m.obs_prob, 1.0 - m.obs_prob)

    return SparseCat(os, probs)
end

function laserbounce(ranges, robot, obstacle)
    left, right, up, down = ranges
    diff = obstacle - robot
    if iszero(diff[1]) && diff[2] > 0
        up = min(up, diff[2] - 1)
    elseif iszero(diff[1]) && diff[2] < 0
        down = min(down, -diff[2] - 1)
    elseif iszero(diff[2]) && diff[1] > 0
        right = min(right, diff[1] - 1)
    elseif iszero(diff[2]) && diff[1] < 0
        left = min(left, -diff[1] - 1)
    end
    return SVector(left, right, up, down)
end

end
