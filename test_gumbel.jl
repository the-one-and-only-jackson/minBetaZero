using Distributed

# addprocs(
#     [("jawa5671@omak.colorado.edu", 1)];
#     dir="/home/jawa5671/",
#     exename="/home/jawa5671/.juliaup/bin/julia",
#     sshflags=`-vvv`
# )

addprocs(20)

@everywhere begin
    using minBetaZero
    using ParticleFilterTrees
    using POMDPs
    using POMDPTools
    using ParticleFilters
    using Flux
    using Statistics
    using Plots
    using ProgressMeter
end

@everywhere include("models/LightDark.jl")
@everywhere using .LightDark

@everywhere function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
    rep = Float32[p.y for p in particles(b)]
    reshape(rep, 1, :)
end

@everywhere mean_std_layer(x) = dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)

params = minBetaZeroParameters(
    GumbelSolver_args = (;
        max_depth           = 10,
        n_particles         = 100,
        tree_queries        = 100,
        max_time            = Inf,
        k_o                 = 20.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 10_000, 
        beliefcache_size    = 10_000,
        m_acts_init         = 3,
        cscale              = 1.,
        cvisit              = 50.
    ),
    t_max           = 50,
    n_episodes      = 20,
    n_iter          = 1000,
    batchsize       = 128,
    lr              = 3e-4,
    lambda          = 1e-3,
    plot_training   = false,
    train_device    = gpu,
    inference_device = gpu,
    buff_cap = 25_000,
    train_intensity = 16,
    warmup_steps = 9_000
)

function f(buff_cap, train_intensity, warmup_steps)
    sum(1 - ((i-1)/i)^train_intensity for i in warmup_steps:buff_cap)
end

f(25_000, 16, 9_000)

nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = 3,
    input_size          = (1,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(-100:10:100),
    p_dropout           = 0.1,
    neurons             = 64,
    hidden_layers       = 2,
    shared_net          = mean_std_layer,
    shared_out_size     = (2,) # must manually set... fix at a later date...        
)

# do multiple epochs and checkpointing

results = []
for _ in 1:5
    @time net, info = betazero(params, LightDarkPOMDP(), ActorCritic(nn_params));

    plot(
        plot(info[:steps], info[:returns]; label=false, xlabel="Steps", title="Mean Episodic Return"),
        plot(info[:episodes], info[:returns]; label=false, xlabel="Episodes");
        layout=(2,1)
    ) |> display

    push!(results, (net, info))
end

p = plot(xlabel="Episodes", title="Mean Episodic Return - $(params.buff_cap) Buffer - $(params.n_episodes) Episodes")
for (_, info) in results
    plot!(p, info[:episodes], info[:returns]; label=false)
end
p

# first plot cscale 1
# second plot cscale 0.1


master_results = deepcopy(results)    



params = minBetaZeroParameters(
    GumbelSolver_args = (;
        max_depth           = 10,
        n_particles         = 500,
        tree_queries        = 100,
        max_time            = Inf,
        k_o                 = 30.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 1_000, 
        beliefcache_size    = 1_000,
        m_acts_init         = 3,
        cscale              = 1.,
        cvisit              = 50.
    ),
    t_max           = 50,
    n_episodes      = 40,
    n_iter          = 500,
    batchsize       = 128,
    lr              = 3e-4,
    lambda          = 0.0,
    plot_training   = false,
    train_device    = gpu,
    inference_device = gpu,
    buff_cap = 40_000,
    n_particles = 500,
)
    

net = results[1][1]
pomdp = LightDarkPOMDP()


data = minBetaZero.test_network(net, pomdp, params; n_episodes=10_000, type=minBetaZero.netPolicyStoch)
histogram(data)
mean(data)
std(data)/sqrt(length(data))

data = minBetaZero.test_network(net, pomdp, params; n_episodes=10_000, type=minBetaZero.netPolicy)
histogram(data)
mean(data)
std(data)/sqrt(length(data))

data = gen_data_distributed(pomdp, net, params; n_episodes=10_000)
histogram(data)
mean(data)
std(data)/sqrt(length(data))



function gen_data_distributed(pomdp, net, params::minBetaZeroParameters; n_episodes=100)
    (; GumbelSolver_args) = params
    
    ret_vec = Float32[]

    data = pmap(1:nworkers()) do i
        function getpolicyvalue(b)
            b_rep = minBetaZero.input_representation(b)
            out = net(b_rep; logits=true)
            return (; value=out.value[], policy=vec(out.policy))
        end
        planner = solve(GumbelSolver(; getpolicyvalue, GumbelSolver_args...), pomdp)

        local_data = []
        for _ in 1:floor(n_episodes/nworkers())
            data = minBetaZero.work_fun(pomdp, planner, params)
            push!(local_data, data)
        end

        return local_data
    end

    ret_vec = Float32[]
    for local_data in data, d in local_data
        push!(ret_vec, d.returns)
    end


    return ret_vec
end



