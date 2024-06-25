struct MDPAgent{S, A, M <: MDP, RNG <: AbstractRNG}
    mdp     :: M
    mcts    :: GumbelSearch{S, A, M, RNG}
    rng     :: RNG
    history :: History{S}
end

function MDPAgent(mdp::MDP; rng::AbstractRNG = Random.default_rng(), kwargs...)
    mcts    = GumbelSearch(mdp; rng, kwargs...)
    state   = rand(rng, initialstate(mdp))
    history = History(mdp, state)
    agent   = MDPAgent(mdp, mcts, rng, history)
    return agent
end

function initialize_agent!(agent::MDPAgent)
    (; mcts, history) = agent
    insert_root!(mcts, history.current_state)
    s_querry = mcts_forward!(mcts)
    return s_querry
end

function agent_main!(agent::MDPAgent, value, logits)
    (; mdp, mcts, rng, history) = agent

    mcts_backward!(mcts, value, logits)

    history_ret = nothing

    if Base.isdone(mcts)
        a , a_info = root_info(mcts)
        sp, reward = @gen(:sp, :r)(mdp, history.current_state, a, rng)

        push!(history;
            sp,
            reward,
            policy_target = a_info.policy_target,
            value_target  = a_info.value_target
        )

        if isterminal(mdp, sp) || length(history.state) >= 250
            calculate_value_target(history)
            newstate = rand(rng, initialstate(mdp))
            history_ret = copy_and_reset!(history, newstate)
        end

        insert_root!(mcts, history.current_state)
    end

    s_querry = mcts_forward!(mcts)

    return s_querry, history_ret
end
