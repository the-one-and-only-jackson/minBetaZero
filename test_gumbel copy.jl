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
    stack(y->convert(SVector{4, Int}, y) / 3f0, particles(b))
end

@everywhere function mean_std_layer(x) 
    mu_x = mean(x; dims=2)
    std_x = std(x; dims=2, mean=mu_x)
    dropdims(cat(mu_x, std_x; dims=1); dims=2)
end

include("testtools.jl")

@everywhere include("cgf.jl")


pomdp = LaserTagPOMDP()
na = length(actions(pomdp))
nx = 4

params = minBetaZeroParameters(
    GumbelSolver_args = (;
        tree_queries        = 20,
        k_o                 = 10.,
        check_repeat_obs    = false,
        resample            = true,
        cscale              = 1.,
        cvisit              = 50.,
        n_particles         = 100, 
        m_acts_init         = 2
    ),
    t_max           = 250,
    n_episodes      = 500,
    inference_batchsize = 128,
    n_iter          = 100,
    batchsize       = 256,
    lr              = 3e-4,
    value_scale     = 0.5,
    lambda          = 1e-3,
    plot_training   = true,
    train_on_planning_b = true,
    n_particles = 10_000,
    n_planning_particles = 100,
    train_device    = gpu,
    train_intensity = 4,
    input_dims = (nx,),
    na = na,
    use_belief_reward = false,
    use_gumbel_target = true,
    num_itr_stored = 1,
    n_net_episodes = 10*nworkers()
)

nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = na,
    input_size          = (nx,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(-100:1:100),
    p_dropout           = 0.1,
    neurons             = 256,
    hidden_layers       = 2,
    shared_net          = mean_std_layer, # Chain(x->clamp.((x .- 5) ./ 2, -10, 10), lightdark_st()), # CGF(1=>64), # mean_std_layer,
    shared_out_size     = (2nx,) # must manually set... fix at a later date...        
)

@time net_moment, info_moment = betazero(params, pomdp, ActorCritic(nn_params));

p = plot(xlabel="Steps", ylabel="Mean Episodic Return", legend=:bottomright, right_margin=4Plots.mm)
bounds = info_moment[:returns] .+ info_moment[:returns_error] .* [-1 1]
error_plot!(p, info_moment[:steps] .- first(info_moment[:steps]), info_moment[:returns], bounds; label="Moments - Tree", c=1)
bounds = info_moment[:network_returns] .+ info_moment[:network_returns_error] .* [-1 1]
error_plot!(p, info_moment[:steps] .- first(info_moment[:steps]), info_moment[:network_returns], bounds; label="Moments - Net", c=2)

plot!(ylims=(0,70), yticks=0:10:70, xlims=(0,0.5e6))
plot!(info_moment[:steps] .- first(info_moment[:steps]), 23*ones(length(info_moment[:steps])); label="QMDP", linestyle=:dash, color=:black)


nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = na,
    input_size          = (nx,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(-100:1:100),
    p_dropout           = 0.1,
    neurons             = 256,
    hidden_layers       = 2,
    shared_net          = CGF(nx=>128), # Chain(x->clamp.((x .- 5) ./ 2, -10, 10), lightdark_st()), # CGF(1=>64), # mean_std_layer,
    shared_out_size     = (128,) # must manually set... fix at a later date...        
)

@time net_cgf, info_cgf = betazero(params, pomdp, ActorCritic(nn_params));

bounds = info_cgf[:returns] .+ info_cgf[:returns_error] .* [-1 1]
error_plot!(p, info_cgf[:steps] .- first(info_moment[:steps]), info_cgf[:returns], bounds; label="CGF - Tree", c=3)
bounds = info_cgf[:network_returns] .+ info_cgf[:network_returns_error] .* [-1 1]
error_plot!(p, info_cgf[:steps] .- first(info_moment[:steps]), info_cgf[:network_returns], bounds; label="CGF - Net", c=4)


savefig("random_state_lasertag.png")

@everywhere include("transformer.jl")

function local_SetTransformer(params::TransformerParams; induced=true)
    embed = [Dense(params.d_in=>params.d)]
    if induced
        encoder = [InducedAttBlock(params.k_enc, params) for _ in 1:params.n_enc]
    else
        encoder = [SelfAttBlock(params) for _ in 1:params.n_enc]
    end
    decoder = [LearnedQuerriesBlock(params.k_dec, params)]
    output = Any[]
    params.prenorm && pushfirst!(output, params.norm(params.d))
    Chain(embed..., encoder..., decoder..., output...)
end

lightdark_st() = Chain(
    x->reshape(x, size(x,1), size(x,2), :),
    local_SetTransformer(TransformerParams(; 
        d_in = 4,
        d = 128, # d = dout because this is a shared layer
        dropout = 0.1,
        n_enc = 2,
        n_dec = 1,
        k_enc = 4,
        k_dec = 2,
    ); induced=false),
    x->(selectdim(x,2,1), selectdim(x,2,2))
)
nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = na,
    input_size          = (nx,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(-100:1:100),
    p_dropout           = 0.1,
    neurons             = 256,
    hidden_layers       = 2,
    shared_net          = lightdark_st(), # Chain(x->clamp.((x .- 5) ./ 2, -10, 10), lightdark_st()), # CGF(1=>64), # mean_std_layer,
    shared_out_size     = (128,2) # must manually set... fix at a later date...        
)

@time net_trans, info_trans = betazero(params, pomdp, ActorCritic(nn_params));


y_rand = -68.4
y_qmdp = 23.0

p = plot(xlabel="Steps", title="Normalized Mean Episodic Return", legend=:bottomright, right_margin=4Plots.mm)

x = info_moment[:steps] .- first(info_moment[:steps])
y1 = info_moment[:returns]
y1_b = info_moment[:returns] .+ info_moment[:returns_error] .* [-1 1]
y2 = info_moment[:network_returns]
y2_b = info_moment[:network_returns] .+ info_moment[:network_returns_error] .* [-1 1]

error_plot!(p, x, (y1 .- y_rand)/(y_qmdp - y_rand), (y1_b .- y_rand)/(y_qmdp - y_rand); label="Moments - Tree", c=1)
error_plot!(p, x, (y2 .- y_rand)/(y_qmdp - y_rand), (y2_b .- y_rand)/(y_qmdp - y_rand); label="Moments - Net", c=2)

x = info_cgf[:steps] .- first(info_cgf[:steps])
y1 = info_cgf[:returns]
y1_b = info_cgf[:returns] .+ info_cgf[:returns_error] .* [-1 1]
y2 = info_cgf[:network_returns]
y2_b = info_cgf[:network_returns] .+ info_cgf[:network_returns_error] .* [-1 1]

error_plot!(p, x, (y1 .- y_rand)/(y_qmdp - y_rand), (y1_b .- y_rand)/(y_qmdp - y_rand); label="CGF - Tree", c=3)
error_plot!(p, x, (y2 .- y_rand)/(y_qmdp - y_rand), (y2_b .- y_rand)/(y_qmdp - y_rand); label="CGF - Net", c=4)


plot!(ylims=(0,1.6), yticks=0:0.3:1.5, xlims=(0,0.5e6))

savefig("random_state_lasertag_normalized.png")



net_1 = deepcopy(net)
info_1 = deepcopy(info)

net_2 = deepcopy(net)
info_2 = deepcopy(info)



    # label = "$n_episodes ep / itr at inten $train_intensity"

using Random

getpolicyvalue = minBetaZero.getpolicyvalue_cpu(net_moment)
solver = GumbelSolver(; getpolicyvalue, params.GumbelSolver_args...)
planner = solve(solver, pomdp)


up = BootstrapFilter(pomdp, 10_000)
b = initialize_belief(up, initialstate(pomdp))
s = rand(initialstate(pomdp))

b_perm = randperm(10000)[1:100]
b_querry = ParticleCollection(particles(b)[b_perm])

a, a_info = action_info(planner, b_querry)
aid = actionindex(pomdp, a)
sp, r, o = @gen(:sp,:r,:o)(pomdp, s, a)

s = sp
b = update(up, b, a, o)

net(input_representation(b_querry))
a_info.Q_root

Dict(x => y for (x,y) in zip(planner.ordered_actions, a_info.tree.b_P[1]))
Dict(x => y for (x,y) in zip(planner.ordered_actions, a_info.policy_target))



test_params = minBetaZeroParameters(
    GumbelSolver_args = (;
        tree_queries        = 10,
        k_o                 = 5.,
        check_repeat_obs    = false,
        resample            = true,
        cscale              = 0.1,
        cvisit              = 50.,
        n_particles         = 100,
        m_acts_init         = 2
    ),
    t_max           = 250,
    n_episodes      = 100,
    inference_batchsize = 128,
    n_iter          = 100,
    batchsize       = 256,
    lr              = 3e-4,
    value_scale     = 1.0,
    lambda          = 1e-3,
    plot_training   = false,
    train_on_planning_b = true,
    n_particles = 10_000,
    n_planning_particles = 500,
    train_device    = gpu,
    train_intensity = 4,
    input_dims = (nx,),
    na = na,
    use_belief_reward = false,
    use_gumbel_target = true,
    num_itr_stored = 5
)


net_results = minBetaZero.test_network(net_cgf, pomdp, test_params; n_episodes=500, policy=minBetaZero.netPolicyStoch)
mean(net_results)
std(net_results)/sqrt(length(net_results))
histogram(net_results; bins=-100:20:100, weights=fill(1/length(net_results), length(net_results)), label=false)
p1 = plot!(ylims=(0,0.5), title="Reward for Stochastic Policy Network")

net_results_max = minBetaZero.test_network(net_cgf, pomdp, test_params; n_episodes=500, policy=minBetaZero.netPolicy)
mean(net_results_max)
histogram(net_results_max; bins=-100:20:100, weights=fill(1/length(net_results_max), length(net_results_max)), label=false)
p2 = plot!(ylims=(0,0.5), title="Reward for Argmax Policy Network")

plot(p1,p2; layout=(1,2), size=(900,450))
savefig("lasertag_histogram.png")

extrema(net_results)


params.t_max