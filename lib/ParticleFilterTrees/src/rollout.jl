struct PlaceHolderUpdater <: Updater end

POMDPs.update(::PlaceHolderUpdater, args...) = error("Updater is placeholder")


struct PORollout
    solver::Solver
    updater::Updater
    rng::AbstractRNG
    n_rollouts::Int # number of rollouts per value estimation. if 0, rollout all particles.
    d::Union{Nothing, Int}
end

function PORollout(sol::Solver, d=10; n_rollouts::Int=1, rng::AbstractRNG=Random.default_rng())
    return PORollout(sol, PlaceHolderUpdater(), rng, n_rollouts, d)
end

struct SolvedPORollout{P<:Policy,U<:Updater,RNG<:AbstractRNG,PMEM<:ParticleCollection}
    policy::P
    updater::U
    rng::RNG
    n_rollouts::Int
    ib::PMEM
    rb::PMEM
    d::Union{Nothing, Int}
end

function MCTS.convert_estimator(est::ParticleFilterTrees.PORollout, sol, pomdp::POMDP)
    upd = est.updater
    if upd isa PlaceHolderUpdater
        upd = PFTFilter(pomdp, sol.n_particles)
    end
    S = statetype(pomdp)
    policy = MCTS.convert_to_policy(est.solver, pomdp)
    ParticleFilterTrees.SolvedPORollout(
        policy,
        upd,
        est.rng,
        est.n_rollouts,
        ParticleCollection(Vector{S}(undef,sol.n_particles)),
        ParticleCollection(Vector{S}(undef,sol.n_particles)),
        est.d
    )
end

# function MCTS.estimate_value(est::BasicPOMCP.SolvedFOValue, pomdp::POMDP{S}, s::S, d::Int) where S
#     POMDPs.value(est.policy, s)
# end

function MCTS.estimate_value(est::ParticleFilterTrees.SolvedPORollout, pomdp::POMDP{S}, b::PFTBelief{S}, d::Int) where S
    b_ = initialize_belief!(est.updater, b, est.ib)
    if est.n_rollouts < 1
        return full_rollout(est, pomdp, b_, d)
    else
        return partial_rollout(est, pomdp, b_, d)
    end
end

function full_rollout(est::ParticleFilterTrees.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, d::Int) where S
    v = 0.0
    b_ = est.rb
    for (s,w) in weighted_particles(b)
        b_.particles .= est.ib.particles
        v += w*rollout(est, pomdp, b_, s, d)
    end
    return v
end

function partial_rollout(est::ParticleFilterTrees.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, d::Int) where S
    v = 0.0
    max_depth = isnothing(est.d) ? d : est.d
    b_ = est.rb
    for _ in 1:est.n_rollouts
        b_.particles .= est.ib.particles
        s = rand(est.rng, b)
        v += rollout(est, pomdp, b_, s, max_depth::Int)
    end
    return v/est.n_rollouts
end

function rollout(est::ParticleFilterTrees.SolvedPORollout, pomdp::POMDP{S}, b::ParticleCollection{S}, s::S, max_depth::Int) where S
    updater = est.updater
    rng = est.rng
    policy = est.policy

    disc = 1.0
    r_total = 0.0
    step = 1

    while !isterminal(pomdp, s) && step ≤ max_depth

        a = ParticleFilters.action(policy, b)

        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a, rng)

        r_total += disc*r

        s = sp

        bp = update(updater, b, a, o)
        b = bp

        disc *= discount(pomdp)
        step += 1
    end

    return r_total
end


struct PFTFilter{PM<:POMDP,RNG<:AbstractRNG,PMEM} <: Updater
    pomdp::PM
    rng::RNG
    p::PMEM # initial and post-resampling particles (p → pp → p)
    w::Vector{Float64}
end

function PFTFilter(pomdp::POMDP, n_p::Int, rng::AbstractRNG)
    S = statetype(pomdp)
    return PFTFilter(
        pomdp,
        rng,
        ParticleCollection(Vector{S}(undef,n_p)),
        Vector{Float64}(undef, n_p)
        )
end

PFTFilter(pomdp::POMDP, n_p::Int) = PFTFilter(pomdp, n_p, Random.default_rng())

function initialize_belief!(pf::PFTFilter, source::PFTBelief, dest::ParticleCollection)
    resample!(source, dest, pf.rng)
end

function initialize_belief(pf::PFTFilter, source::PFTBelief{S}) where S
    return initialize_belief!(pf, source, ParticleCollection(Vector{S}(undef)))
end

"""
predict!
    - propagate b(up.p) → up.p
reweight!
    - update up.w
    - s ∈ b(up.p), sp ∈ up.p
resample!
    - resample up.p → b
"""
function update!(up::PFTFilter, b::ParticleCollection, a, o)
    predict!(up.p, up.pomdp, b, a, up.rng) # b → up.p
    reweight!(up.w, up.pomdp, b, a, up.p.particles, o)
    resample!(up.p, up.w, b, up.rng) # up.p → b
end

POMDPs.update(up::PFTFilter, b::ParticleCollection, a, o) = update!(up,b,a,o)

function predict!(pm::ParticleCollection, m::POMDP, b::ParticleCollection, a, rng::AbstractRNG)
    all_terminal = true
    pm_particles = pm.particles
    for i in 1:n_particles(b)
        s = particle(b, i)
        if !isterminal(m, s)
            all_terminal = false
            sp = @gen(:sp)(m, s, a, rng)
            @inbounds pm_particles[i] = sp
        end
    end
    # all_terminal && @warn "All particles terminal in internal filter"
    return all_terminal
end

function resample!(b::ParticleCollection, w::Vector{Float64}, bp::ParticleCollection, rng::AbstractRNG)
    n_p = n_particles(b)
    ws = sum(w)
    ps = bp.particles

    r = rand(rng)*ws/n_p
    c = w[1]
    i = 1
    U = r
    for m in 1:n_p
        while U > c && i < n_p
            i += 1
            c += w[i]
        end
        U += ws/n_p
        @inbounds ps[m] = b.particles[i]
    end
    return bp
end

function resample!(b::PFTBelief, bp::ParticleCollection, rng::AbstractRNG)
    n_p = n_particles(b)
    w = b.weights
    ws = sum(w)
    ps = bp.particles

    r = rand(rng)*ws/n_p
    c = w[1]
    i = 1
    U = r
    for m in 1:n_p
        while U > c && i < n_p
            i += 1
            c += w[i]
        end
        U += ws/n_p
        ps[m] = b.particles[i]
    end
    return bp
end