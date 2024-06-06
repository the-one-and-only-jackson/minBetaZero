# using POMDPs
# import POMDPTools:RandomPolicy
# using Random
# include("LaserTag.jl")

module LaserTag

using POMDPs, Random, StaticArrays, POMDPTools, POMDPModelTools

export LTState, LaserTagPOMDP, DiscreteLaserTagPOMDP

struct LTState{S,T}
    robot::S
    target::T
end

Base.convert(::Type{SVector{4, Int}}, s::LTState) = SA[s.robot..., s.target...]
Base.convert(::Type{AbstractVector{Int}}, s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractVector}, s::LTState) = convert(SVector{4, Int}, s)
Base.convert(::Type{AbstractArray}, s::LTState) = convert(SVector{4, Int}, s)

struct LaserTagPOMDP{S} <: POMDP{LTState, Symbol, SVector{5,Int}}
    size::SVector{2, Int}
    obstacles::Set{SVector{2, Int}}
    blocked::BitArray{2}
    robot_init::S
    obsindices::Array{Union{Nothing,Int}, 5}
end

const DiscreteLaserTagPOMDP = LaserTagPOMDP{SVector{2, Int64}}

function lasertag_observations(size)
    os = SVector{5,Int}[]
    for same_grid in 0:1, left in 0:size[1]-1, right in 0:size[1]-left-1, up in 0:size[2]-1, down in 0:size[2]-up-1
        push!(os, SVector(same_grid, left, right, up, down))
    end
    return os
end

#=
***************************************************************************************
LaserTag POMDp with discrete robot state space
=#
function DiscreteLaserTagPOMDP(; 
    size=(10, 7), n_obstacles=9, rng::AbstractRNG=Random.MersenneTwister(20)
    )

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

    obsindices = Array{Union{Nothing,Int}}(nothing, 2, size[1], size[1], size[2], size[2])
    for (ind, o) in enumerate(lasertag_observations(size))
        obsindices[(o.+1)...] = ind
    end

    LaserTagPOMDP(SVector(size), obstacles, blocked, robot_init, obsindices)
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

function POMDPs.states(m::DiscreteLaserTagPOMDP)
    vec(collect(LTState(SVector(c[1],c[2]), SVector(c[3], c[4])) for c in Iterators.product(1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2])))
end

POMDPs.actions(::DiscreteLaserTagPOMDP) = collect(keys(actionind))
POMDPs.observations(m::DiscreteLaserTagPOMDP) = lasertag_observations(m.size)
POMDPs.discount(m::DiscreteLaserTagPOMDP) = 0.997
POMDPs.stateindex(m::DiscreteLaserTagPOMDP, s) = LinearIndices((1:m.size[1], 1:m.size[2], 1:m.size[1], 1:m.size[2]))[s.robot..., s.target...]
POMDPs.actionindex(::DiscreteLaserTagPOMDP, a) = actionind[a]
POMDPs.obsindex(m::DiscreteLaserTagPOMDP, o) = m.obsindices[(o.+1)...]::Int
POMDPs.isterminal(::DiscreteLaserTagPOMDP, s) = s.robot == s.target


function POMDPs.initialstate(m::LaserTagPOMDP)
    states = LTState[]
    for x in 1:m.size[1],y in 1:m.size[2]
        target_state = SVector(x,y)
        if(!(target_state in m.obstacles))
            push!(states,LTState(m.robot_init,target_state))
        end
    end
    return Uniform(states)
end

function POMDPs.transition(m::LaserTagPOMDP, s, a)

    oldrobot = s.robot
    newrobot = move_robot(m, oldrobot, a)

    if isterminal(m, s)
        @assert s.robot == s.target
        # return a new terminal state where the robot has moved
        # this maintains the property that the robot always moves the same, 
        # regardless of the target state
        return SparseCat([LTState(newrobot, newrobot)], [1.0])
    end

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
    elseif sp.robot in sp.target
        return 100.0
    elseif a == :measure
        return -2.0
    else
        return -1.0
    end
end

function POMDPs.reward(m::LaserTagPOMDP, s, a)
    r = 0.0
    td = transition(m, s, a)
    #weighted_iterator is a function provided by POMDPTools.jl
    for (sp, w) in weighted_iterator(td)
        r += w*reward(m, s, a, sp)
    end
    return r
end

function POMDPs.gen(m::DiscreteLaserTagPOMDP,s::LTState,a::Symbol,rng::AbstractRNG)

    curr_robot = s.robot
    curr_target = s.target

    #Move Robot
    new_robot = move_robot(m,curr_robot,a)

    #Move Target
    T_target = target_transition_likelihood(m,curr_robot,new_robot,curr_target)
    new_target = rand(rng, T_target)

    #Get Observation
    Z_target = observation_likelihood(m,a,new_robot,new_target)
    o = rand(rng,Z_target)

    sp = LTState(new_robot,new_target)
    r = reward(m,s,a,sp)

    return (; sp, o, r)
end

function Base.in(s::StaticVector{2,Int}, o::StaticVector{2,Int})
    return s[1]==o[1] && s[2]==o[2]
end

function Base.in(s::StaticVector{2,Float64}, o::StaticVector{2,Int})
    grid_x = floor(Int, s[1])
    grid_y = floor(Int, s[2])
    return SVector(grid_x,grid_y) == o
end

function Base.in(s::StaticVector{2,Float64}, o::Set{SVector{2, Int}})
    grid_x = floor(Int, s[1])
    grid_y = floor(Int, s[1])
    return SVector(grid_x, grid_y) in o
end

function bounce(m, pos, change)
    # The dot operator in clamp below specifies that apply the clamp operation to each
    # entry of that SVector with corresponding lower and upper bounds
    new = clamp.(pos + change, SVector(1,1), m.size)
    return m.blocked[new[1], new[2]] ? pos : new
end

function check_collision(m,old_pos,new_pos)
    l = LineSegment(old_pos,new_pos)
    delta_op = (SVector(0, 1), SVector(1, 0))
    delta_corner = (SVector(-1, 0), SVector(0, -1))

    for o in m.obstacles
        for delta in delta_op
            obs_boundary = LineSegment(o, o + delta)
            !isempty(intersection(l, obs_boundary)) && return true
        end
        corner_point = o + SVector(1,1)
        for delta in delta_corner
            obs_boundary = LineSegment(corner_point, corner_point + delta)
            !isempty(intersection(l, obs_boundary)) && return true
        end
    end

    return false
end

function move_robot(m, pos::StaticVector{2,<:AbstractFloat}, a::AbstractVector)
    new_pos = pos + clamp.(a,-1,1) # a == change
    if( new_pos[1] >= 1.0+m.size[1] || new_pos[1] < 1.0 ||
        new_pos[2] >= 1.0+m.size[2] || new_pos[2] < 1.0 ||
        check_collision(m,pos,new_pos) )
        return pos
    else
        return new_pos
    end
end

function move_robot(m, pos::StaticVector{2,<:AbstractFloat}, a::Symbol)
    return a==:measure ? pos : move_robot(m, pos, actiondir[a])
end

move_robot(m, pos::StaticVector{2,<:Integer}, a::Symbol) = bounce(m, pos, actiondir[a])

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
    if(robot_pos in target_pos)
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