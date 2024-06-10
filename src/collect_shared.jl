function collect_data(buffer::DataBuffer, storage_channel, n_episodes::Int)
    progress = Progress(n_episodes)
    ret_vec = zeros(Float32, n_episodes)
    steps = 0

    for i in 1:n_episodes
        data = take!(storage_channel)
        steps += length(data.value_target)
        set_buffer!(
            buffer; 
            network_input = data.network_input, 
            value_target  = data.value_target, 
            policy_target = data.policy_target
        )
        ret_vec[i] = data.returns
        next!(progress)
    end
    
    close(storage_channel)

    return ret_vec, steps
end

function collect_data_returns(storage_channel, n_episodes::Int)
    progress = Progress(n_episodes)
    ret_vec = zeros(Float32, n_episodes)
    steps = 0

    for i in 1:n_episodes
        data = take!(storage_channel)
        steps += length(data.value_target)
        ret_vec[i] = data.returns
        next!(progress)
    end
    
    close(storage_channel)

    return ret_vec, steps
end

function work_fun(pomdp, planner, params)
    (; t_max, n_particles, n_planning_particles, train_on_planning_b, use_belief_reward, use_gumbel_target) = params

    use_gumbel_target = use_gumbel_target && isa(planner, GumbelPlanner)

    up = BootstrapFilter(pomdp, n_particles)
    b = initialize_belief(up, initialstate(pomdp))
    s = rand(initialstate(pomdp))

    b_vec = []
    aid_vec = Int[]
    gumbel_target_vec = Vector[]
    state_reward = Float32[]
    belief_reward = Float32[]

    for _ in 1:t_max
        if n_planning_particles == n_particles
            b_querry = b
        else
            b_perm = randperm(n_particles)[1:n_planning_particles]
            b_querry = ParticleCollection(particles(b)[b_perm])
        end
        
        a, a_info = action_info(planner, b_querry)
        aid = actionindex(pomdp, a)
        s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)

        b_target = train_on_planning_b ? b_querry : b

        push!.((b_vec, aid_vec, state_reward), (b_target, aid, r))

        if use_gumbel_target
            push!(gumbel_target_vec, a_info.policy_target)
        end

        if use_belief_reward
            br = 0f0
            for p in particles(b)
                br += reward(pomdp, p, a) / n_particles
            end
            push!(belief_reward, br)
        end
        
        if isterminal(pomdp, s)
            break
        end

        b = POMDPs.update(up, b, a, o)
    end

    gamma = discount(pomdp)

    for i in length(state_reward)-1:-1:1
        state_reward[i] += gamma * state_reward[i+1]
    end

    if use_belief_reward
        for i in length(belief_reward)-1:-1:1
            belief_reward[i] += gamma * belief_reward[i+1]
        end
    end

    output_reward = use_belief_reward ? belief_reward : state_reward

    if use_gumbel_target
        policy_target = reduce(hcat, gumbel_target_vec)
    else
        policy_target = Flux.onehotbatch(aid_vec, 1:length(actions(pomdp)))
    end

    data = (; 
        network_input = stack(input_representation, b_vec), 
        value_target  = reshape(output_reward, 1, :), 
        policy_target = policy_target, 
        returns       = state_reward[1]
    )

    return data
end