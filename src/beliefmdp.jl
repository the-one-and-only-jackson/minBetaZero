
# for now discrete actions only
struct ParticleBeliefMDP{P<:POMDP, B<:ParticleFilters.AbstractParticleBelief, A} <: MDP{B, Int}
    pomdp::P
    ordered_actions::A
    _actions::Base.OneTo{Int}
end
function ParticleBeliefMDP(pomdp::POMDP{S,A}) where {S,A}
    _ordered_actions = ordered_actions(pomdp)
    _actions = Base.OneTo(length(_ordered_actions))
    ParticleBeliefMDP{typeof(pomdp), PFTBelief{S}, typeof(_ordered_actions)}(pomdp, _ordered_actions, _actions)
end

POMDPs.actions(bmdp::ParticleBeliefMDP) = bmdp._actions
POMDPs.actions(bmdp::ParticleBeliefMDP, b) = bmdp._actions # update later

POMDPs.actionindex(::ParticleBeliefMDP, a::Int) = a

POMDPs.isterminal(::ParticleBeliefMDP, b) = ParticleFilterTrees.isterminalbelief(b)

POMDPs.discount(bmdp::ParticleBeliefMDP) = discount(bmdp.pomdp)

function POMDPs.initialstate(bmdp::ParticleBeliefMDP)
    POMDPModelTools.ImplicitDistribution() do rng
        ParticleFilterTrees.initialize_belief(rng, bmdp.pomdp, initialstate(bmdp.pomdp), 500)
    end
end

function POMDPs.gen(bmdp::ParticleBeliefMDP, b, a_idx, rng::AbstractRNG)
    a = bmdp.ordered_actions[a_idx]
    p_idx = ParticleFilterTrees.non_terminal_sample(rng, bmdp.pomdp, b)
    sample_s = ParticleFilters.particle(b, p_idx)
    sample_sp, o, sample_r = @gen(:sp,:o,:r)(bmdp.pomdp, sample_s, a, rng)
    bp, weighted_return, non_terminal_ws = ParticleFilterTrees.GenBelief(
        rng, bmdp.pomdp, b, a, o, p_idx, sample_sp, sample_r, true
    )
    return (sp=bp, r=weighted_return, info=non_terminal_ws)
end
