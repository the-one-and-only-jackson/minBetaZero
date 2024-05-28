##
#  Action selection
# 
#  MCTS.convert_estimator(::Type{BasicActionSelector}, sol::PFTDPWSolver, pomdp::POMDP)
##

abstract type PFTActionSelector end

select_action(selector, tree, b_idx, d) = select_action(selector, tree, b_idx)


struct BasicActionSelector{F, P<:POMDP, R<:AbstractRNG} <: PFTActionSelector
    k_a::Float64
    alpha_a::Float64
    enable_action_pw::Bool
    criterion::F
    rng::R
    pomdp::P
end

function BasicActionSelector(sol::PFTDPWSolver, pomdp::POMDP)
    BasicActionSelector(
        sol.k_a,
        sol.alpha_a,
        sol.enable_action_pw,
        sol.criterion,
        sol.rng,
        pomdp
    )
end

function select_action(criteria::BasicActionSelector, tree::PFTDPWTree, b_idx)
    (;k_a, alpha_a, enable_action_pw, rng, pomdp, criterion) = criteria

    if enable_action_pw && length(tree.b_children[b_idx]) ≤ k_a*tree.Nh[b_idx]^alpha_a
        a = rand(rng, pomdp)
        insert_action!(tree, b_idx, a)
    elseif !enable_action_pw && isempty(tree.b_children[b_idx])
        for a in actions(pomdp)
            insert_action!(tree, b_idx, a)
        end
    end

    select_action(criterion, tree, b_idx)
end

## BasicActionSelector criteria

@kwdef struct MaxUCB
    c::Float64 = 1.0
end

function select_action(criteria::MaxUCB, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    Nh = tree.Nh[b_idx]
    b_children = tree.b_children[b_idx]
    Nh < length(b_children) && return b_children[Nh + 1]

    c = criteria.c
    lnh = log(Nh)
    local opt_a::A
    max_ucb = -Inf
    opt_idx = 0

    for (a,ba_idx) in b_children
        Nha = tree.Nha[ba_idx]
        iszero(Nha) && return a => ba_idx
        Q̂ = tree.Qha[ba_idx]
        ucb = Q̂ + c*sqrt(lnh / Nha)

        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a => opt_idx
end

@kwdef struct MaxPoly
    c::Float64 = 1.0
    β::Float64 = 1/4
end

function select_action(criteria::MaxPoly, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    Nh = tree.Nh[b_idx]
    b_children = tree.b_children[b_idx]
    Nh < length(b_children) && return b_children[Nh + 1]

    (;c,β) = criteria
    powNh = Nh^β
    local opt_a::A
    max_ucb = -Inf
    opt_idx = 0

    for (a,ba_idx) in b_children
        Nha = tree.Nha[ba_idx]
        iszero(Nha) && return a => ba_idx
        Q̂ = tree.Qha[ba_idx]
        ucb = Q̂ + c*powNh / sqrt(Nha)

        if ucb > max_ucb
            max_ucb = ucb
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a => opt_idx
end

struct MaxQ end

function select_action(::MaxQ, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    local opt_a::A
    maxQ = -Inf
    opt_idx = 0

    for (a,ba_idx) in tree.b_children[b_idx]
        Q̂ = tree.Qha[ba_idx]

        if Q̂ > maxQ
            maxQ = Q̂
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a => opt_idx
end

#=
# fix this later, this should be root action selection
# this code is for non-root action selection
struct Gumbel{P,F}
    pomdp::P
    get_policy::F
end

function select_best(criteria::Gumbel, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    Nh = tree.Nh[b_idx]
    b_children = tree.b_children[b_idx]

    local opt_a::A
    max_target = -Inf
    opt_idx = 0

    policy = criteria.get_policy(tree.b[b_idx])

    for (a,ba_idx) in b_children
        Nha = tree.Nha[ba_idx]
        target = policy[POMDPs.actionindex(criteria.pomdp, a)] - Nha/Nh

        if target > max_target
            max_target = target
            opt_a = a
            opt_idx = ba_idx
        end
    end
    return opt_a => opt_idx
end
=#

struct ConstantDefaultAction{A}
    a::A
end

function (da::ConstantDefaultAction)(pomdp::POMDP, ::Any)
    @warn "Default PFT-DPW action"
    return da.a
end

struct RandomDefaultAction end

function (da::RandomDefaultAction)(pomdp::POMDP, ::Any)
    @warn "Default PFT-DPW action"
    return rand(actions(pomdp))
end




struct FastRandomSolver
    rng::Random.AbstractRNG
    d::Union{Nothing, Int}
end

FastRandomSolver(d=nothing) = FastRandomSolver(Random.default_rng(), d)

struct FastRandomRolloutEstimator{A, RNG <:AbstractRNG}
    actions::A
    rng::RNG
    d::Union{Nothing, Int}
end

function FastRandomRolloutEstimator(pomdp::POMDP, estim::FastRandomSolver)
    RNG = typeof(estim.rng)
    act = actions(pomdp)
    A = typeof(act)
    return FastRandomRolloutEstimator{A,RNG}(act, estim.rng, estim.d)
end

POMDPs.action(p::FastRandomRolloutEstimator, ::Any) = rand(p.rng, p.actions)

function MCTS.convert_estimator(estimator::FastRandomSolver, ::Any, pomdp::POMDP)
    return FastRandomRolloutEstimator(pomdp, estimator)
end

function sr_gen(estimator::FastRandomRolloutEstimator, pomdp::POMDP{S,A}, s::S, a::A) where {S,A}
    return sr_gen(estimator.rng, pomdp, s, a)
end

function MCTS.estimate_value(estimator::FastRandomRolloutEstimator, pomdp::POMDP{S}, s::S, max_depth::Int) where S
    disc = 1.0
    r_total = 0.0
    rng = estimator.rng
    step = 1

    while !isterminal(pomdp, s) && step ≤ max_depth
        a = action(estimator, s)
        sp,r = sr_gen(estimator, pomdp, s, a)
        r_total += disc*r
        s = sp
        disc *= discount(pomdp)
        step += 1
    end

    return r_total
end

function MCTS.estimate_value(est, pomdp::POMDP{S}, b::PFTBelief{S}, d::Int) where S
    v = 0.0
    max_depth = isnothing(est.d) ? d : est.d
    for (s,w) in weighted_particles(b)
        v += w*MCTS.estimate_value(est, pomdp, s, max_depth::Int)
    end
    return v
end

