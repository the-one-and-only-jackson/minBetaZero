function is_obs_required(pomdp::POMDP)
    s,sp = rand(initialstate(pomdp)), rand(initialstate(pomdp))
    a = rand(actions(pomdp))

    obs_req = false
    try # very crude. would like to use `applicable` or `hasmethod` but these fail with quickpomdps
        POMDPs.reward(pomdp, s, a, sp)
    catch e
        obs_req = true
    end

    return obs_req
end

function sr_gen(::Val{false}, rng::AbstractRNG, pomdp::POMDP{S,A}, s::S, a::A) where {S,A}
    sp = rand(rng, transition(pomdp, s, a))
    r = reward(pomdp, s, a, sp)
    return sp, r
end

function sr_gen(::Val{true}, rng::AbstractRNG, pomdp::POMDP{S,A}, s::S, a::A) where {S,A}
    return @gen(:sp,:r)(pomdp, s, a, rng)
end
