@kwdef mutable struct SeqHalf
    const n::Int # sim budget
    const m::Int # initial number of actions
    N::Int = 0  # target number of sims per action
    k::Int = 0 # halving parameter
end

function next!(sh::SeqHalf)
    dN = (2 ^ sh.k) * sh.n / (sh.m * ceil(log2(sh.m)))
    sh.k += 1
    sh.N += max(1, floor(Int, dN))
    return sh.N
end

function reset!(sh::SeqHalf)
    sh.k = 0
    sh.N = 0
    next!(sh)
end
