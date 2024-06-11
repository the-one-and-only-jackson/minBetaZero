using Distributed

# addprocs(10)

@everywhere begin
    using minBetaZero
    using POMDPs
    using POMDPTools
    using ParticleFilters
    using Flux
    using Statistics
end

@everywhere include("models/LightDark.jl")
@everywhere using .LightDark

@everywhere function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
    rep = Float32[p.y for p in particles(b)]
    reshape(rep, 1, :)
end

@everywhere mean_std_layer(x) = dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)

include("testtools.jl")

@everywhere include("cgf.jl")

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
        d_in = 1,
        d = 64, # d = dout because this is a shared layer
        dropout = 0.1,
        n_enc = 2,
        n_dec = 1,
        k_enc = 4,
        k_dec = 2,
    ); induced=false),
    x->(selectdim(x,2,1), selectdim(x,2,2))
)


using CUDA

function testfun()
    x = randn(Float32, 1, 100, 32)
    x = x |> gpu
    net = lightdark_st()
    net = net |> gpu
    @benchmark $net($x)
end

math_mode = CUDA.PEDANTIC_MATH
math_mode = CUDA.DEFAULT_MATH
math_mode = CUDA.FAST_MATH

precision = :Float16
precision = :BFloat16
precision = :TensorFloat32

CUDA.math_mode!(math_mode; precision)
testfun()

x = cu(randn(500,500,32));
y = cu(randn(500,500,32));
z = cu(zeros(500,500,32));

CUDA.math_mode!(CUDA.FAST_MATH; precision = :Float16)
CUDA.@profile raw=true batched_mul!(z, x, y)

CUDA.math_mode!(CUDA.DEFAULT_MATH; precision = :TensorFloat32)
CUDA.@profile raw=true batched_mul!(z, x, y)

CUDA.math_mode!(CUDA.PEDANTIC_MATH; precision = :TensorFloat32)
CUDA.@profile raw=true batched_mul!(z, x, y)

# do multiple epochs and checkpointing

p1 = plot(ylabel="MCTS Return", title="Mean Episodic Return", ylims=(0,18), legend=:bottomright, xlabel="Episodes", xlims=(0, 5_000), right_margin=4Plots.mm)
p2 = plot(ylabel="Network Return", xlabel="Episodes", ylims=(-5,20), legend=:bottomright)

for n_episodes in [500, 100], train_intensity in [8, 4]
    params = minBetaZeroParameters(
        GumbelSolver_args = (;
            tree_queries        = 40,
            k_o                 = 20.,
            check_repeat_obs    = false,
            resample            = true,
            cscale              = 1.0,
            cvisit              = 50.,
            n_particles         = 100,
            m_acts_init         = 3    
        ),
        t_max           = 50,
        n_episodes      = n_episodes,
        inference_batchsize = 32,
        n_iter          = 5_000 ÷ n_episodes,
        batchsize       = 128,
        lr              = 3e-4,
        value_scale     = 1.0,
        lambda          = 1e-3,
        plot_training   = false,
        train_on_planning_b = true,
        n_particles = 1_000,
        n_planning_particles = 100,
        train_device    = gpu,
        train_intensity = train_intensity,
        input_dims = (1,),
        na = 3,
        use_belief_reward = true,
        use_gumbel_target = true,
        on_policy = true
    )

    nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
        action_size         = 3,
        input_size          = (1,),
        critic_loss         = Flux.Losses.logitcrossentropy,
        critic_categories   = collect(-100:10:100),
        p_dropout           = 0.1,
        neurons             = 256,
        hidden_layers       = 1,
        shared_net          = Chain(Flux.Scale(1), lightdark_st()), # Chain(x->clamp.((x .- 5) ./ 2, -10, 10), lightdark_st()), # CGF(1=>64), # mean_std_layer,
        shared_out_size     = (64,2) # must manually set... fix at a later date...        
    )

    results = []
    for _ in 1:3
        @time net, info = betazero(params, LightDarkPOMDP(), ActorCritic(nn_params));

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
    savefig("figures/lightdark_transformer.png")
end



# p = plot(xlabel="Episodes", title="Mean Episodic Return - $(params.buff_cap) Buffer - $(params.n_episodes) Episodes")
# for (_, info) in results
#     plot!(p, info[:episodes], info[:returns]; label=false)
# end
# p


pomdp = LightDarkPOMDP()
net = ActorCritic(nn_params)
n_particles = 1_000

up = BootstrapFilter(pomdp, n_particles)
b = initialize_belief(up, initialstate(pomdp))
s = rand(initialstate(pomdp))

b_vec = []
aid_vec = Int[]
gumbel_target_vec = Vector[]
state_reward = Float32[]
belief_reward = Float32[]

getpolicyvalue = minBetaZero.getpolicyvalue_cpu(net)
solver = GumbelSolver(; getpolicyvalue, params.GumbelSolver_args...)
planner = solve(solver, pomdp)

b_perm = randperm(n_particles)[1:100]
b_querry = ParticleCollection(particles(b)[b_perm])
a, a_info = action_info(planner, b_querry)
aid = actionindex(pomdp, a)
s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)
b = update(up, b, a, o)

a_info.tree.b_V





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
    batchsize       = 256,
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



