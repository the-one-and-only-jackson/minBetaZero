struct PFTBelief{T} <: AbstractParticleBelief{T}
    particles::Vector{T}
    weights::Vector{Float64}
    non_terminal_ws::Float64
end

@inline ParticleFilters.n_particles(b::PFTBelief) = length(b.particles)
@inline ParticleFilters.particles(p::PFTBelief) = p.particles
ParticleFilters.weighted_particles(b::PFTBelief) = (p=>w for (p,w) in zip(particles(b), weights(b)))
@inline ParticleFilters.weight_sum(b::PFTBelief) = 1.0 # gaurentee this elsewhere... greedy
@inline ParticleFilters.weight(b::PFTBelief, i::Int) = b.weights[i]
@inline ParticleFilters.particle(b::PFTBelief, i::Int) = b.particles[i]
@inline ParticleFilters.weights(b::PFTBelief) = b.weights

function Random.rand(rng::AbstractRNG, sampler::Random.SamplerTrivial{<:PFTBelief})
    b = sampler[]
    t = rand(rng) * weight_sum(b)
    N = n_particles(b)
    i = 1
    cw = weight(b,1)
    while cw < t && i < N
        i += 1
        @inbounds cw += weight(b,i)
    end
    return particle(b,i)
end

function non_terminal_sample(rng::AbstractRNG, pomdp::POMDP, b::PFTBelief)
    @assert b.non_terminal_ws > eps()
    t = rand(rng) * b.non_terminal_ws
    N = n_particles(b)
    i = 1
    cw = isterminal(pomdp,particle(b,1)) ? 0.0 : weight(b,1)
    while cw < t && i < N
        i += 1
        isterminal(pomdp,particle(b,i)) && continue
        @inbounds cw += weight(b,i)
    end
    return i
end

function resample!(rng::AbstractRNG, b::PFTBelief{T}, ps::Vector{T}) where T
    n_p = n_particles(b)
    @assert length(ps) >= n_p "length(ps) == $(length(ps)), np=$n_p"

    ws = weight_sum(b)
    U = rand(rng)*ws/n_p
    @inbounds c = weight(b,1)
    i = 1

    for m in 1:n_p
        while U > c && i < n_p
            i += 1
            @inbounds c += weight(b,i)
        end
        U += ws/n_p
        @inbounds ps[m] = particle(b,i)
    end

    copyto!(particles(b), view(ps, 1:n_p))
    fill!(weights(b), inv(n_p))
    
    return b
end

@inline isterminalbelief(b::PFTBelief; tol=eps()) = b.non_terminal_ws < tol

function initialize_belief(rng, pomdp::POMDP{S}, b, n) where {S}
    initialize_belief!(rng, Vector{S}(undef,n), Vector{Float64}(undef,n), pomdp, b)
end

function initialize_belief!(
    rng::AbstractRNG, s::Vector{S}, w::Vector{Float64}, pomdp::POMDP{S}, b
    ) where {S}

    w_i = inv(length(s))
    fill!(w, w_i) # all particles equal weight
    rand!(rng, s, b)
    non_terminal_ws = w_i * count(x->!isterminal(pomdp,x), s)
    return PFTBelief(s, w, non_terminal_ws)
end

function initialize_belief!(
    rng::AbstractRNG, s::Vector{S}, w::Vector{Float64}, pomdp::POMDP{S}, b::PFTBelief
    ) where {S}

    fill!(w, inv(length(s))) # all particles equal weight
    for i in eachindex(s)
        s[i] = particle(b, non_terminal_sample(rng,pomdp,b))
    end
    return PFTBelief(s, w, 1.0)
end

function initialize_belief!(
    rng::AbstractRNG, s::Vector{S}, w::Vector{Float64}, pomdp::POMDP{S}, b::ParticleCollection
    ) where {S}

    fill!(w, inv(length(s))) # all particles equal weight

    perm = randperm(rng, n_particles(b))
    j = 1
    for i in eachindex(s)
        for k in 1:n_particles(b)
            s[i] = particle(b, perm[j])
            j = mod1(j+1, n_particles(b))

            if !isterminal(pomdp, s[i])
                break
            end

            if k == n_particles(b)
                @warn "All particles at root are terminal"
                return PFTBelief(s, w, 0.0)
            end
        end
    end

    return PFTBelief(s, w, 1.0)
end

#= 
function initialize_belief!(
    rng::AbstractRNG, s::Vector{S}, w::Vector{Float64}, pomdp::POMDP{S}, b
    ) where {S}
    
    w_i = inv(length(s))
    w = fill!(w,w_i) # all particles equal weight
    terminal_count = 0

    if b isa AbstractParticleBelief
        perm = randperm(n_particles(b))
        perm_idx = 0
        for s_idx in eachindex(s)
            perm_idx += 1
            p = particle(b, perm[perm_idx])
            terminal_flag = isterminal(pomdp, p)
            while terminal_flag && perm_idx < n_particles(b)
                perm_idx += 1
                p = particle(b, perm[perm_idx])
                terminal_flag = isterminal(pomdp, p)
            end
            if !terminal_flag
                s[s_idx] = p
            elseif s_idx == 1
                @warn "All states in input belief are terminal"
                @views rand!(rng, s, b)
                terminal_count = length(s)
                break
            else
                @warn "Fewer non-terminal beliefs (N = $(s_idx-1)) than PFT particles \\
                    (N = $(length(s)))"
                @views rand!(rng, s[s_idx:end], s[1:s_idx-1])
                break
            end
        end
    else
        rand!(rng, s, b)
        terminal_count = count(x->isterminal(pomdp,x), s)
    end

    non_terminal_ws = 1 - w_i * terminal_count
    return PFTBelief(s, w, non_terminal_ws)
end
=#


function GenBelief(
    rng::AbstractRNG,
    pomdp::POMDP{S,A,O},
    b::PFTBelief{S},
    a::A,
    o::O,
    p_idx::Int,
    sample_sp::S,
    sample_r::Float64,
    resample::Bool
    ) where {S,A,O}

    N = n_particles(b)
    bp_particles = Vector{S}(undef, N)
    bp_weights = Vector{Float64}(undef, N)
    resample_cache = Vector{S}(undef, N)

    GenBelief(
        rng, bp_particles, bp_weights, resample_cache,
        pomdp, b, a, o, p_idx, sample_sp, sample_r, resample
    )
end

function GenBelief(
    rng::AbstractRNG,
    bp_particles::Vector{S}, # empty vec
    bp_weights::Vector{Float64}, # empty vec
    resample_cache::Vector{S}, # empty vec
    pomdp::POMDP{S,A,O},
    b::PFTBelief{S},
    a::A,
    o::O,
    p_idx::Int,
    sample_sp::S,
    sample_r::Float64,
    resample::Bool;
    tol::Float64 = eps()
    ) where {S,A,O}

    weighted_return = 0.0
    non_terminal_ws_raw = 0.0 # will be unnormalized
    terminal_ws     = 0.0 # either 0 or sum of those particle's previous weight
    n_nt = 0

    flag = false

    for (i,(s,w)) in enumerate(weighted_particles(b))
        # Propagation
        if i === p_idx
            (sp, r) = sample_sp, sample_r
            flag = true
        elseif isterminal(pomdp, s)
            (sp, r) = (s, 0.0)
        else
            (sp, r) = sr_gen(rng, pomdp, s, a) # @gen(:sp,:r)(pomdp, s, a, rng)
        end

        weighted_return += r * w
        bp_particles[i] = sp

        if isterminal(pomdp, sp)
            bp_weights[i] = resample ? 0.0 : w
            terminal_ws += w
        else
            bp_weights[i] = w * pdf(POMDPs.observation(pomdp, s, a, sp), o)
            non_terminal_ws_raw += bp_weights[i]
            n_nt += 1
        end    
    end

    allterminal = n_nt > 0 && non_terminal_ws_raw <= tol

    non_terminal_ws = 1.0 - terminal_ws

    # normalize weights
    if !allterminal
        # reweight non-terminal particles
        normalizing_factor = non_terminal_ws/non_terminal_ws_raw
        for (i, p) in enumerate(bp_particles)
            isterminal(pomdp, p) && continue
            bp_weights[i] *= normalizing_factor
        end
    end

    if resample && !allterminal
        bp = PFTBelief(bp_particles, bp_weights, 1.0)
        resample!(rng, bp, resample_cache)    
    else
        bp = PFTBelief(bp_particles, bp_weights, non_terminal_ws)
    end

    # discount future returns in belief mdp if resampling
    nt_prob = resample ? non_terminal_ws : 1.0 

    return bp::PFTBelief{S}, weighted_return::Float64, nt_prob::Float64
end

function sr_gen(rng::AbstractRNG, pomdp::P, s::S, a::A) where {S,A,P<:POMDP{S,A}}
    hasmethod(reward, Tuple{P,S,A,S}) || return @gen(:sp,:r)(pomdp, s, a, rng)

    sp = rand(rng, transition(pomdp, s, a))
    r = reward(pomdp, s, a, sp)
    return sp, r
end