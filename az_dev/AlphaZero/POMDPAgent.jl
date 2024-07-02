mutable struct POMDPAgent
    const agent     :: MDPAgent
    const pomdp     :: POMDP
    const updater   :: POMDPs.Updater
    const rng       :: AbstractRNG
    state           :: S
end

function POMDPAgent(pomdp::POMDP, updater::POMDPs.Updater, bmdp::MDP)

    agent = MDPAgent()

    return POMDPAgent(agent, pomdp, updater, rng, state)
end

initialize_agent!(agent::POMDPAgent) = initialize_agent!(agent.agent)

function step_agent!(agent::POMDPAgent, history_channel::Channel)
    (; mdp, mcts, rng, history, max_steps, segment_length) = agent

    s = history.state[end]
    a, a_info = root_info(mcts)
    sp, r = @gen(:sp, :r)(mdp, s, a, rng)

    push!(history.reward, r)
    push!(history.policy_target, a_info.policy_target)
    # push!(history.value_target, a_info.value_target)

    history.episode_reward += r * discount(mdp) ^ history.steps
    history.steps += 1

    terminated = isterminal(mdp, sp)
    truncated  = history.steps >= max_steps

    if terminated || truncated
        sp = rand(rng, initialstate(mdp))
        history.trajectory_done = true
    end

    if terminated || truncated || length(history) == segment_length
        v_end = terminated ? zero(a_info.next_value) : a_info.next_value
        calculate_value_target(history, v_end; discount = discount(mdp))

        history_copy = copy_and_reset!(history)
        put!(history_channel, history_copy)
    end

    push!(history.state, sp)
    insert_root!(mcts, sp)

    return nothing
end
