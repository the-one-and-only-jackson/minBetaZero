using POMDPs, Random, StaticArrays, POMDPTools, SparseArrays

include("models/lasertag.jl")
using .LaserTag

function randsp(pomdp, T, os, s, a)
    si = stateindex(pomdp, s)
    Ta = T[a]
    spi = rowvals(Ta)
    irange = nzrange(Ta, si)
    p = nonzeros(Ta)

    w = zero(eltype(p))
    r = rand(eltype(p))
    i = first(irange)
    while i < last(irange)
        w += p[i]
        if w < r
            i += 1
        else
            break
        end
    end
    
    sp = os[spi[i]]

    return sp
end

function test()
    pomdp = LaserTagPOMDP()
    T = transition_matrices(pomdp; sparse=true)
    for a in actions(pomdp)
        T[a] = copy(transpose(T[a]))
    end
    os = ordered_states(pomdp)
    s = rand(initialstate(pomdp))
    a = rand(actions(pomdp))

    randsp(pomdp, T, os, s, a)
end