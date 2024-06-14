module LaserTag

using POMDPs, Random, StaticArrays, POMDPTools

export LTState, LaserTagPOMDP

struct LTState
    robot::SVector{2, Int}
    target::SVector{2, Int}
end

Base.convert(::Type{SVector{4, Int}}    , s::LTState) = SA[s.robot..., s.target...]
Base.convert(::Type{AbstractVector{Int}}, s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractVector}     , s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractArray}      , s::LTState) = convert(SVector{4, Int}, s)

struct LaserTagPOMDP{S} <: POMDP{LTState, Symbol, SVector{5,Int}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    initial_states::S
    obsindices::Array{Union{Nothing,Int}, 5}
end

lasertag_observations(size) = [
    SVector(same_grid, left, right, up, down)
    for same_grid in 0:1
    for left      in 0:size[1]-1
    for right     in 0:size[1]-1-left
    for up        in 0:size[2]-1
    for down      in 0:size[2]-1-up
]



#=
***************************************************************************************
LaserTag POMDp with discrete robot state space
=#
function DiscreteLaserTagPOMDP(;size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20))
    obstacles = Set{SVector{2, Int}}()
    blocked = falses(size...)
    while length(obstacles) < n_obstacles
        obs = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
        push!(obstacles, obs)
        blocked[obs...] = true
    end

    #Initialize Robot
    robot_init = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    while robot_init in obstacles
        robot_init = SVector(rand(rng, 1:size[1]), rand(rng, 1:size[2]))
    end

    initial_states = Uniform([
        LTState(robot_init, SVector(x, y))
        for x in 1:size[1]
        for y in 1:size[2]
        if !in(SVector(x,y), obstacles)
    ])

    obsindices = Array{Union{Nothing,Int}}(nothing, 2, size[1], size[1], size[2], size[2])
    for (ind, o) in enumerate(lasertag_observations(size))
        obsindices[(o.+1)...] = ind
    end

    LaserTagPOMDP(SVector(size), obstacles, blocked, initial_states, obsindices)
end

const actionind = Dict(
    :left=>1, :right=>2, :up=>3, :down=>4, :left_up=>5, :right_up=>6, :left_down=>7, 
    :right_down=>8, :measure=>9
)

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

POMDPs.initialstate(m::LaserTagPOMDP) = m.initial_states
POMDPs.actions(::LaserTagPOMDP) = collect(keys(actionind))
POMDPs.observations(m::LaserTagPOMDP) = lasertag_observations(m.size)
POMDPs.discount(m::LaserTagPOMDP) = 0.99
POMDPs.actionindex(::LaserTagPOMDP, a) = actionind[a]
POMDPs.obsindex(m::LaserTagPOMDP, o) = m.obsindices[(o .+ 1)...]
POMDPs.isterminal(::LaserTagPOMDP, s) = s.robot == s.target

function POMDPs.stateindex(m::LaserTagPOMDP, s)
    idxs = (1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2])
    LinearIndices(idxs)[s.robot..., s.target...]
end


function POMDPs.transition(m::LaserTagPOMDP, s, a)
    if isterminal(m, s)
        return Deterministic(s)
    end

    oldrobot = s.robot
    newrobot = bounce(m, oldrobot, actiondir[a])

    oldtarget = s.target
    target_T = target_transition_likelihood(m,oldrobot,newrobot,oldtarget)

    states = LTState[]
    probs = Float64[]
    for i in eachindex(target_T.vals)
        newtarget = target_T.vals[i]
        push!(states,LTState(newrobot,newtarget))
        push!(probs,target_T.probs[i])
    end

    return SparseCat(states, probs)
end

function POMDPs.observation(m::LaserTagPOMDP, a, sp)
    R = sp.robot
    T = sp.target
    O_likelihood = observation_likelihood(m,a,R,T)
    return O_likelihood
end

function POMDPs.reward(m::LaserTagPOMDP, s, a, sp)
    if isterminal(m, s)
        return 0.0
    elseif sp.robot == sp.target
        return 100.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end

function bounce(m::LaserTagPOMDP, pos::SVector{2, Int}, change::SVector{2, Int})
    newpos = clamp.(pos + change, SVector(1,1), m.size)
    isblocked = m.blocked[newpos...]
    return isblocked ? pos : newpos
end


function check_collision(m,old_pos,new_pos)
    l = LineSegment(old_pos,new_pos)
    delta_op = ( SVector(0,1),SVector(1,0) )
    delta_corner = ( SVector(-1,0),SVector(0,-1) )

    for o in m.obstacles
        for delta in delta_op
            obs_boundary = LineSegment(o,o+delta)
            # println(l,obs_boundary)
            if( !isempty(intersection(l,obs_boundary)) )
                return true
            end
        end
        corner_point = o+SVector(1,1)
        for delta in delta_corner
            obs_boundary = LineSegment(corner_point,corner_point+delta)
            # println(l,obs_boundary)
            if( !isempty(intersection(l,obs_boundary)) )
                return true
            end
        end
    end
    return false
end

function target_transition_likelihood(m,oldrobot_pos,newrobot_pos,oldtarget)
    targets = [oldtarget]
    targetprobs = Float64[0.0]
    newrobot = SVector( Int(floor(newrobot_pos[1])),Int(floor(newrobot_pos[2])) )
    if sum(abs, newrobot - oldtarget) > 2 # move randomly
        for change in (SVector(-1,0), SVector(1,0), SVector(0,1), SVector(0,-1))
            newtarget = bounce(m, oldtarget, change)
            if newtarget == oldtarget
                targetprobs[1] += 0.25
            else
                push!(targets, newtarget)
                push!(targetprobs, 0.25)
            end
        end
    else # move away
        oldrobot = SVector( Int(floor(oldrobot_pos[1])),Int(floor(oldrobot_pos[2])) )
        away = sign.(oldtarget - oldrobot)
        if sum(abs, away) == 2 # diagonal
            away = away - SVector(0, away[2]) # preference to move in x direction
        end
        newtarget = bounce(m, oldtarget, away)
        targets[1] = newtarget
        targetprobs[1] = 1.0
    end

    target_states = SVector{2,Int}[]
    probs = Float64[]
    for (t, tp) in zip(targets, targetprobs)
        push!(target_states, t)
        push!(probs, tp)
    end

    return SparseCat(target_states, probs)
end

function laserbounce(ranges, robot, obstacle)
    left, right, up, down = ranges
    diff = obstacle - robot
    if diff[1] == 0
        if diff[2] > 0
            up = min(up, diff[2]-1)
        elseif diff[2] < 0
            down = min(down, -diff[2]-1)
        end
    elseif diff[2] == 0
        if diff[1] > 0
            right = min(right, diff[1]-1)
        elseif diff[1] < 0
            left = min(left, -diff[1]-1)
        end
    end
    return SVector(left, right, up, down)
end

function observation_likelihood(m, a, newrobot, target_pos)
    robot_pos = SVector( Int(floor(newrobot[1])),Int(floor(newrobot[2])) )
    left = robot_pos[1]-1
    right = m.size[1]-robot_pos[1]
    up = m.size[2]-robot_pos[2]
    down = robot_pos[2]-1
    ranges = SVector(left, right, up, down)
    for obstacle in m.obstacles
        ranges = laserbounce(ranges, robot_pos, obstacle)
    end
    ranges = laserbounce(ranges, robot_pos, target_pos)
    if robot_pos == target_pos
        os = SVector(SVector(1,ranges...), SVector(1, 0, 0, 0, 0))
    else
        os = SVector(SVector(0,ranges...), SVector(0, 0, 0, 0, 0))
    end
    # os = SVector(ranges, SVector(0, 0, 0, 0))
    if all(ranges.==0.0) || a == :measure
        probs = SVector(1.0, 0.0)
    else
        probs = SVector(0.1, 0.9)
    end
    return SparseCat(os, probs)
end



end