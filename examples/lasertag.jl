using Distributed

addprocs(10; topology=:master_worker, enable_threaded_blas=true) # uses around 2GB of memory per worker

@everywhere begin
    using minBetaZero
    using POMDPs
    using POMDPTools
    using ParticleFilters
    using Flux
    using Statistics
end

@everywhere include("../models/lasertag.jl")
@everywhere using .LaserTag

@everywhere begin
    function minBetaZero.input_representation(b::AbstractParticleBelief{<:LTState})
        stack(y->convert(AbstractVector, y) / 3f0, particles(b))
    end

    function minBetaZero.input_representation(b::AbstractVector{<:LTState})
        stack(y->convert(AbstractVector, y) / 3f0, b)
    end

    function mean_std_layer(x) 
        mu_x = mean(x; dims=2)
        std_x = std(x; dims=2, mean=mu_x)
        dropdims(cat(mu_x, std_x; dims=1); dims=2)
    end    
end

include("testtools.jl")

function ffres(d::Int, act=identity; p=0.0, k=4)
    block = Chain(Dense(d, k*d, act), Dense(k*d, d), Flux.Dropout(p))
    return Flux.Parallel(+, identity, block)
end

pomdp = LaserTagPOMDP(; use_measure = false, obs_prob = 0.1)
na = length(actions(pomdp))
nx = 4

params = minBetaZeroParameters(
    GumbelSolver_args = (;
        tree_queries        = 40,
        k_o                 = 20.,
        check_repeat_obs    = false,
        resample            = true,
        cscale              = 0.1,
        cvisit              = 50.,
        n_particles         = 200, 
        m_acts_init         = 2
    ),
    t_max           = 250,
    n_episodes      = 200,
    n_iter          = 200,
    batchsize       = 1024,
    lr              = 1e-3,
    value_scale     = 0.5,
    lambda          = 1e-2,
    plot_training   = true,
    train_on_planning_b = true,
    n_planning_particles = 200,
    train_device    = gpu,
    train_intensity = 8,
    input_dims = (nx,),
    na = na,
    use_belief_reward = false,
    use_gumbel_target = true,
    num_itr_stored = 5,
    n_net_episodes = 200
)

moments_nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = na,
    input_size          = (nx,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(range(-100, 100, length=128)),
    p_dropout           = 0.1,
    neurons             = 256,
    hidden_layers       = 2,
    shared_net          = mean_std_layer,
    shared_out_size     = (2nx,) # must manually set... fix at a later date...        
)

cgf_nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = na,
    input_size          = (nx,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(range(-100, 100, length=128)),
    p_dropout           = 0.1,
    neurons             = 256,
    hidden_layers       = 2,
    shared_net          = CGF(nx=>256), 
    shared_out_size     = (256,) # must manually set... fix at a later date...        
)

# despot 48

@time net_moment, info_moment = betazero(params, pomdp, ActorCritic(moments_nn_params));

p = plot(xlabel="Steps", title="Mean Episodic Return", legend=:bottomright, right_margin=4Plots.mm)

x1 = info_moment[:steps] .- first(info_moment[:steps])
y1 = info_moment[:returns]
y2 = info_moment[:network_returns]

plot_smoothing!(p, x1, y1; k=5, label="Moments - Tree", c=1)
plot_smoothing!(p, x1, y2; k=5, label="Moments - Net", c=2)


@time net_cgf, info_cgf = betazero(params, pomdp, ActorCritic(cgf_nn_params));

x2 = info_cgf[:steps] .- first(info_cgf[:steps])
y3 = info_cgf[:returns]
y4 = info_cgf[:network_returns]

plot_smoothing!(p, x2, y3; label="CGF - Tree", c=3)
plot_smoothing!(p, x2, y4; label="CGF - Net", c=4)

plot!(ylims=(0,50), xlims=(0,max(maximum.((x1,x2))...)), right_margin=5Plots.mm)
savefig("no_measure.png")

y_rand = -68.4 # recalculate for given pomdp
y_qmdp = 23.0 # recalculate for given pomdp

p = plot(xlabel="Steps", title="Normalized Mean Episodic Return", legend=:bottomright, right_margin=4Plots.mm)
plot_smoothing!(p, x1, (y1 .- y_rand)/(y_qmdp - y_rand); label="Moments - Tree", c=1)
plot_smoothing!(p, x1, (y2 .- y_rand)/(y_qmdp - y_rand); label="Moments - Net", c=2)
plot_smoothing!(p, x2, (y3 .- y_rand)/(y_qmdp - y_rand); label="CGF - Tree", c=3)
plot_smoothing!(p, x2, (y4 .- y_rand)/(y_qmdp - y_rand); label="CGF - Net", c=4)
plot!(ylims=(0,1.5), yticks=0:0.3:1.5, xlims=(0,max(maximum.((x1,x2))...)))






net_results = [
    minBetaZero.test_network(net_moment, pomdp, params; n_episodes=100, policy=minBetaZero.netPolicyStoch)
    for _ in 1:100
]

data = mean.(net_results)
mean(data)
std(data)/sqrt(length(data))
histogram(data; label=false, title="100 Episode Mean", xlabel="Mean Return", ylabel="Frequency", weights=fill(1/length(data), length(data)))
savefig("dist.png")
