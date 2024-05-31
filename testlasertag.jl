using minBetaZero

using POMDPs
using POMDPTools
using ParticleFilters
using Flux
using Statistics
using Plots

include("lasertag.jl")

actionindex(pomdp, actions(pomdp)[1])

function minBetaZero.input_representation(b::AbstractParticleBelief{<:LTState})
    stack(y->convert(SVector{4, Int}, y), particles(b))
end

mean_std_layer(x) = dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)

params = minBetaZeroParameters(
    PFTDPWSolver_args = (;
        max_depth           = 10,
        n_particles         = 100,
        tree_queries        = 100,
        max_time            = Inf,
        k_o                 = 24.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 1_000, 
        beliefcache_size    = 1_000,
    ),
    t_max = 100,
    n_episodes = 512,
    n_workers = 16,
    n_iter = 20,
    train_frac = 0.8,
    batchsize = 256,
    lr = 10e-4,
    lambda = 0.0,
    n_epochs = 50,
    plot_training = true,
    train_dev = gpu,
    early_stop = 10,
    n_warmup = 500
)

pomdp = DiscreteLaserTagPOMDP()

nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size=length(actions(pomdp)),
    input_size=(4,),
    critic_loss = Flux.Losses.logitcrossentropy,
    critic_categories = collect(-100:10:100),
    p_dropout = 0.1,
    neurons = 128,
    hidden_layers = 2,
    shared_net = mean_std_layer,
    shared_out_size = (2*4,) # must manually set... fix at a later date...        
)


@time net, info = betazero(params, pomdp, ActorCritic(nn_params));

plot(
    plot(info[:steps], info[:returns]; label=false, xlabel="Steps", title="Mean Episodic Return"),
    plot(info[:episodes], info[:returns]; label=false, xlabel="Episodes");
    layout=(2,1)
)

planner = solve(
    PFTDPWSolver(;
        getpolicyvalue = function(x)
            out = net(input_representation(x); logits=true)
            return (; value=out.value[], policy=vec(out.policy))
        end,
        params.PFTDPWSolver_args...
    ),
    pomdp
)

up = BootstrapFilter(pomdp, 500)

ret_mcts = []

b = initialize_belief(up, initialstate(pomdp))
s = rand(initialstate(pomdp))
s_vec = []
b_vec = []
r_vec = []
for _ in 1:100
    a, a_info = action_info(planner, b)
    sp, r, o = @gen(:sp, :r, :o)(pomdp, s, a)
    push!.((b_vec,r_vec,s_vec), (b,r,s))
    b = update(up, b, a, o)
    s = sp
    isterminal(pomdp, s) && break
end
push!(ret_mcts, r_vec' * discount(pomdp) .^ (0:length(r_vec)-1))

mean(ret_mcts)
std(ret_mcts)/sqrt(length(ret_mcts))


ret_net = []

b = initialize_belief(up, initialstate(pomdp))
s = rand(initialstate(pomdp))
s_vec = []
b_vec = []
r_vec = []
for _ in 1:100
    a = actions(pomdp)[argmax(net(input_representation(b)).policy)]
    sp, r, o = @gen(:sp, :r, :o)(pomdp, s, a)
    push!.((b_vec,r_vec,s_vec), (b,r,s))
    b = update(up, b, a, o)
    s = sp
    isterminal(pomdp, s) && break
end
push!(ret_net, r_vec' * discount(pomdp) .^ (0:length(r_vec)-1))

mean(ret_net)
std(ret_net)/sqrt(length(ret_net))










