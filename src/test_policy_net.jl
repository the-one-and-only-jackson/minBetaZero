struct RandPlanner{A} <: Policy
    actions::A
end
RandPlanner(p::POMDP) = RandPlanner(actions(p))
POMDPs.action(p::RandPlanner, b) = rand(p.actions)

struct netPolicy{N,A} <: Policy
    net::N
    ordered_actions::A
end
POMDPs.action(p::netPolicy, b) = p.ordered_actions[argmax(getlogits(p.net, b))]

struct netPolicyStoch{N,A} <: Policy
    net::N
    ordered_actions::A
end
POMDPs.action(p::netPolicyStoch, b) = p.ordered_actions[argmax(getlogits(p.net, b) + rand(Gumbel(), length(p.ordered_actions)))]

getlogits(net::ActorCritic, b) = vec(net(input_representation(b); logits=true).policy)

function test_network(net, pomdp::POMDP, params::minBetaZeroParameters; n_episodes=500, policy=netPolicyStoch)
    if nprocs() > 1
        distributed_test_network(net, pomdp, params; n_episodes, policy)
    else
        threaded_test_network(net, pomdp, params; n_episodes, policy)
    end
end

function distributed_test_network(net, pomdp::POMDP, params::minBetaZeroParameters; n_episodes, policy)
    ret_vec = pmap(1:n_episodes) do _
        planner = policy(net, ordered_actions(pomdp))
        data = work_fun(pomdp, planner, params)
        return data.returns
    end
end

function threaded_test_network(net, pomdp::POMDP, params::minBetaZeroParameters; n_episodes, policy)
    # single threaded for now
    @warn "Single threaded network test not implemented" maxlog=1
    return [0.0]
end