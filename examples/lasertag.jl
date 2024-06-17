using Distributed

addprocs(12; topology=:master_worker, enable_threaded_blas=true) # uses around 2GB of memory per worker

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

@everywhere struct RMSNorm{S, B}
    scale::S
    bias::B
end
@everywhere begin
    RMSNorm(d; bias=true) = RMSNorm(ones(Float32,d), bias ? zeros(Float32,d) : false)
    (block::RMSNorm)(x) = block.scale .* x .* sqrt.(size(x,1) ./ (sum(abs2, x; dims=1) .+ eps(eltype(x)))) .+ block.bias
    Flux.@layer :expand RMSNorm
end


function ffres(d::Int, act=identity; p=0.0, k=4)
    block = Chain(RMSNorm(d), Dense(d, k*d, act), Dense(k*d, d), Flux.Dropout(p))
    return Flux.Parallel(+, identity, block)
end

pomdp = LaserTagPOMDP(; use_measure = false, obs_prob = 0.1)
na = length(actions(pomdp))
nx = 4

params = minBetaZeroParameters(
    GumbelSolver_args = (;
        tree_queries        = 10,
        k_o                 = 5., # half tree_queries for 2 initial actions and 1-step lookahead
        resample            = true, # force true, evenly weighted particles ideal for training
        cscale              = 0.1, # 1.0 trains faster but less stable
        cvisit              = 50., # not worth tuning
        n_particles         = 100,
        m_acts_init         = 2 # number of actions sampled at root
    ),
    t_max           = 250,
    n_episodes      = 100, # per iteration
    n_iter          = 500,
    batchsize       = 1024,
    lr              = 1e-3,
    value_scale     = 0.5, # learning rate scaling
    lambda          = 1e-2, # weight decay
    plot_training   = false,
    train_on_planning_b = true, # this should probably not be a parameter, force true
    n_planning_particles = 100, # match n_particles in Gumbel args
    train_device    = gpu,
    train_intensity = 8, # 4-8 seems safe?
    input_dims = (nx,), # pomdp dependent input dimension
    na = na, # pomdp dependent number of actions
    use_belief_reward = false,
    use_gumbel_target = true, # policy target
    n_net_episodes = 25, # policy-only evalutions per iteration

    buff_cap        = 25_000,
    warmup_steps    = 5_000
)

moments_nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = na,
    input_size          = (nx,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(range(-100, 100, length=128)),
    p_dropout           = 0.1,
    neurons             = 128,
    hidden_layers       = 1,
    shared_net          = Chain(mean_std_layer, Dense(2nx=>128), ffres(128, gelu; p=0.1, k=4)),
    shared_out_size     = (128,) # must manually set... fix at a later date...
)

cgf_nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = na,
    input_size          = (nx,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(range(-100, 100, length=128)),
    p_dropout           = 0.1,
    neurons             = 128,
    hidden_layers       = 2,
    shared_net          = CGF(nx=>128),
    shared_out_size     = (128,) # must manually set... fix at a later date...
)

# despot 48

@time net_moment, info_moment = betazero(params, pomdp, ActorCritic(moments_nn_params));

p = plot(xlabel="Steps", title="Mean Episodic Return", legend=:bottomright, right_margin=4Plots.mm)

x1 = info_moment[:steps] .- first(info_moment[:steps])
y1 = info_moment[:returns]
y2 = info_moment[:network_returns]

plot_smoothing!(p, x1, y1; k=5, label="Moments - Tree", c=1)
plot_smoothing!(p, x1, y2; k=5, label="Moments - Net", c=2)

y12_max = 5*ceil(maximum(max(maximum(y1), maximum(y2)))/5)

plot!(ylims=(0,y12_max), yticks=0:5:y12_max, xlims=(0,maximum(x1)), right_margin=5Plots.mm)

savefig("examples/moments_3.png")


@time net_cgf, info_cgf = betazero(params, pomdp, ActorCritic(cgf_nn_params));

x2 = info_cgf[:steps] .- first(info_cgf[:steps])
y3 = info_cgf[:returns]
y4 = info_cgf[:network_returns]

plot_smoothing!(p, x2, y3; k=5, label="CGF - Tree", c=3)
plot_smoothing!(p, x2, y4; k=5, label="CGF - Net", c=4)

y34_max = 5*ceil(maximum(max(maximum(y3), maximum(y4)))/5)

plot!(ylims=(0,max(y12_max,y34_max)), yticks=0:5:max(y12_max,y34_max), xlims=(0,max(maximum.((x1,x2))...)), right_margin=5Plots.mm)

savefig("examples/moments_cgf_2.png")

y_rand = -68.4 # recalculate for given pomdp
y_qmdp = 23.0 # recalculate for given pomdp

p = plot(xlabel="Steps", title="Normalized Mean Episodic Return", legend=:bottomright, right_margin=4Plots.mm)
plot_smoothing!(p, x1, (y1 .- y_rand)/(y_qmdp - y_rand); label="Moments - Tree", c=1)
plot_smoothing!(p, x1, (y2 .- y_rand)/(y_qmdp - y_rand); label="Moments - Net", c=2)
plot_smoothing!(p, x2, (y3 .- y_rand)/(y_qmdp - y_rand); label="CGF - Tree", c=3)
plot_smoothing!(p, x2, (y4 .- y_rand)/(y_qmdp - y_rand); label="CGF - Net", c=4)
plot!(ylims=(0,1.5), yticks=0:0.3:1.5, xlims=(0,max(maximum.((x1,x2))...)))



## Showing the distribution of the mean return after 100 episodes

net_results = [
    minBetaZero.test_network(net_moment, pomdp, params; n_episodes=100, policy=minBetaZero.netPolicyStoch)
    for _ in 1:100
]

data = mean.(net_results)
mean(data)
std(data)/sqrt(length(data))
histogram(data; label=false, title="100 Episode Mean", xlabel="Mean Return", ylabel="Frequency", weights=fill(1/length(data), length(data)))
savefig("dist.png")







planner = minBetaZero.netPolicyStoch(net_moment, ordered_actions(pomdp))

up = DiscreteUpdater(pomdp)
b = initialize_belief(up, initialstate(pomdp))
s = rand(initialstate(pomdp))

b_vec = []
b_querry_vec = []
a_vec = []
s_vec = [s]
r_vec = Float64[]

for step_num in 1:250
    b_querry = rand(b, 100)

    a = action(planner, b_querry)
    s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)

    push!.((b_vec, b_querry_vec, a_vec, s_vec, r_vec), (b, b_querry, a, s, r))

    if isterminal(pomdp, s)
        break
    end

    b = POMDPs.update(up, b, a, o)
end

r_total = sum(r_vec .* discount(pomdp) .^ (0:length(r_vec)-1))
println("Epsidoic Return: $r_total")


f(x) = max(0, 1+log10(x)/3)
common = (label=false, seriestype=:scatter, markercolor=:black, markersize=10, xticks=(1.5:10.5, 1:10), yticks=(1.5:7.5, 1:7), xlims=(1,11), ylims=(1,8))
robot_states = [s.robot for s in b_vec[1].state_list]
target_states = [s.target for s in b_vec[1].state_list]

anim = Plots.@animate for i in eachindex(b_vec)
    r = [0.0 for i in 1:10, j in 1:7]
    for (s, p) in zip(robot_states, b_vec[i].b)
        r[s...] += p
    end

    t = [0.0 for i in 1:10, j in 1:7]
    for (s, p) in zip(target_states, b_vec[i].b)
        t[s...] += p
    end

    hr = heatmap(0.5 .+ (1:10), 0.5 .+ (1:7), f.(r)'; c = Plots.cgrad(:roma), clim=(0,1))
    ht = heatmap(0.5 .+ (1:10), 0.5 .+ (1:7), f.(t)'; c = Plots.cgrad(:roma), clim=(0,1))

    obs = stack([pomdp.obstacles...])
    robot_pos = s_vec[i].robot
    target_pos = s_vec[i].target

    for p in [hr, ht]
        plot!(p, [0.5 + robot_pos[1]], [0.5 + robot_pos[2]]; markershape=:circle, common...)
        plot!(p, [0.5 + target_pos[1]], [0.5 + target_pos[2]]; markershape=:star5, common...)
        plot!(p, 0.5 .+ obs[1,:], 0.5 .+ obs[2,:]; markershape=:x, common...)
    end

    plot!(hr, colorbar=nothing, title="Robot Belief")
    plot!(ht, colorbar=nothing, title="Target Belief")

    plot(hr, ht; layout=(1,2), size=(800,400))
end

gif(anim, "examples/belief.gif", fps = 1)
