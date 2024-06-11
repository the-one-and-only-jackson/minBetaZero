using Distributed

addprocs(10)

@everywhere begin
    using minBetaZero
    using POMDPs
    using POMDPTools
    using ParticleFilters
    using Flux
    using Statistics
    using StaticArrays
end

@everywhere include("models/lasertag.jl")
@everywhere using .LaserTag

@everywhere function minBetaZero.input_representation(b::AbstractParticleBelief{<:LTState})
    stack(y->convert(SVector{4, Int}, y), particles(b))
end

@everywhere function mean_std_layer(x) 
    mu_x = mean(x; dims=2)
    @assert all(isfinite, mu_x) "$x"
    std_x = std(x; dims=2, mean=mu_x)
    @assert all(isfinite, std_x) "$x"
    dropdims(cat(mu_x, std_x; dims=1); dims=2)
end


include("testtools.jl")



pomdp = DiscreteLaserTagPOMDP()
up = BootstrapFilter(pomdp, 1_000)
b = initialize_belief(up, initialstate(pomdp))
length(unique(particles(b)))

# @everywhere include("cgf.jl")

p1 = plot(ylabel="MCTS Return", title="Mean Episodic Return", ylims=(0,18), legend=:bottomright, xlabel="Episodes", xlims=(0, 5_000), right_margin=4Plots.mm)

for n_episodes in [500, 100], train_intensity in [4, 8]
    pomdp = DiscreteLaserTagPOMDP()
    na = length(actions(pomdp))
    nx = 4

    params = minBetaZeroParameters(
        GumbelSolver_args = (;
            tree_queries        = 100,
            k_o                 = 10.,
            check_repeat_obs    = false,
            resample            = true,
            cscale              = 1.0,
            cvisit              = 50.,
            n_particles         = 100,
            m_acts_init         = na
        ),
        t_max           = 100,
        n_episodes      = n_episodes,
        inference_batchsize = 32,
        n_iter          = 10_000 ÷ n_episodes,
        batchsize       = 128,
        lr              = 3e-4,
        value_scale     = 1.0,
        lambda          = 1e-3,
        plot_training   = true,
        train_on_planning_b = true,
        n_particles = 1_000,
        n_planning_particles = 100,
        train_device    = gpu,
        train_intensity = train_intensity,
        input_dims = (nx,),
        na = na,
        use_belief_reward = true,
        use_gumbel_target = true,
        on_policy = true
    )

    nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
        action_size         = na,
        input_size          = (nx,),
        critic_loss         = Flux.Losses.logitcrossentropy,
        critic_categories   = collect(-100:10:100),
        p_dropout           = 0.1,
        neurons             = 256,
        hidden_layers       = 1,
        shared_net          = Chain(mean_std_layer, Flux.Scale(2*nx)), # Chain(x->clamp.((x .- 5) ./ 2, -10, 10), lightdark_st()), # CGF(1=>64), # mean_std_layer,
        shared_out_size     = (2*nx,) # must manually set... fix at a later date...        
    )

    results = []
    for _ in 1:3
        @time net, info = betazero(params, pomdp, ActorCritic(nn_params));

        plot(
            plot(info[:steps], info[:returns]; label=false, xlabel="Steps", title="Mean Episodic Return"),
            plot(info[:episodes], info[:returns]; label=false, xlabel="Episodes");
            layout=(2,1)
        ) |> display

        push!(results, (net, info))
    end 


    label = "$n_episodes ep / itr at inten $train_intensity"

    x = results[1][2][:episodes] .- first(results[1][2][:episodes])

    y = stack(x->x[2][:returns], results)
    mu, bounds = ci_bounds(y)
    error_plot!(p1, x, mu, bounds; label)

    plot(p1) |> display
    savefig("figures/lasertag.png")
end



