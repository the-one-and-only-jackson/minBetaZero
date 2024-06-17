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

    # up = BootstrapFilter(pomdp, n_particles)
    up = DiscreteUpdater(pomdp)
    b = initialize_belief(up, initialstate(pomdp))
    s = rand(initialstate(pomdp))

    b_vec = []
    aid_vec = Int[]
    gumbel_target_vec = Vector[]
    state_reward = Float32[]
    belief_reward = Float32[]
    tree_value_target = Float32[]

    for step_num in 1:t_max
        # if n_planning_particles == n_particles
        #     b_querry = b
        # else
        #     b_perm = randperm(n_particles)[1:n_planning_particles]
        #     b_querry = ParticleCollection(particles(b)[b_perm])
        # end
        b_querry = rand(b, n_planning_particles)

        a, a_info = action_info(planner, b_querry)
        aid = actionindex(pomdp, a)
        s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)

        b_target = train_on_planning_b ? b_querry : b

        push!.((b_vec, aid_vec, state_reward), (b_target, aid, r))

        if use_gumbel_target
            push!(gumbel_target_vec, a_info.policy_target)
            push!(tree_value_target, a_info.v_mix)
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

        # b = myupdate(up, b, a, o)
        b = POMDPs.update(up, b, a, o)

        if isnothing(b) || (b isa AbstractParticleBelief && any(isterminal(pomdp, p) for p in particles(b)))
            if isnothing(b)
                @warn "Particle depletion - random rollout"
            else
                @warn "Terminal particle in belief - random rollout"
            end

            for t in 1:t_max-step_num
                a = rand(actions(pomdp))
                s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)
                state_reward[end] += discount(pomdp)^t * r
                if use_belief_reward
                    belief_reward[end] += discount(pomdp)^t * r
                end
                if isterminal(pomdp, s)
                    break
                end
            end

            break
        end

        # if step_num == t_max
        #     r = net(input_representation(b)).value[]
        #     state_reward[end] += discount(pomdp) * r
        #     belief_reward[end] += discount(pomdp) * r
        # end
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
        # value_target = reshape(tree_value_target, 1, :)
        value_target  = reshape(output_reward, 1, :)
    else
        policy_target = Flux.onehotbatch(aid_vec, 1:length(actions(pomdp)))
        value_target  = reshape(output_reward, 1, :)
    end

    data = (;
        network_input = stack(input_representation, b_vec),
        value_target  = value_target,
        policy_target = policy_target,
        returns       = state_reward[1]
    )

    return data
end


function myupdate(up::BasicParticleFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    resize!(pm, n_particles(b))
    resize!(wm, n_particles(b))
    ParticleFilters.predict!(pm, up.predict_model, b, a, o, up.rng)
    ParticleFilters.reweight!(wm, up.reweight_model, b, a, pm, o, up.rng)

    w_sum = sum(wm)

    return w_sum < eps() ? nothing : resample(
        up.resampler,
        WeightedParticleBelief(pm, wm, w_sum, nothing),
        up.predict_model,
        up.reweight_model,
        b, a, o,
        up.rng
    )
end

Base.eltype(::Type{DiscreteBelief{P,T}}) where {P,T} = T
function Random.rand(rng::Random.AbstractRNG, st::Random.SamplerTrivial{<:DiscreteBelief})
    b = st[]
    i = sample(rng, Weights(b.b))
    return b.state_list[i]
end
