using Distributed

# addprocs(
#     [("jawa5671@omak.colorado.edu", 1)];
#     dir="/home/jawa5671/",
#     exename="/home/jawa5671/.juliaup/bin/julia",
#     sshflags=`-vvv`
# )

addprocs(2)

@everywhere begin
    using minBetaZero
    using POMDPs
    using POMDPTools
    using ParticleFilters
    using Flux
    using Statistics
    using Plots
    using ProgressMeter
    using StaticArrays
end

using POMDPs, POMDPTools

pomdp = cpp_emu_lasertag(4)

@everywhere using LaserTag

@everywhere function minBetaZero.input_representation(b::AbstractParticleBelief{LTState})
    stack(y->convert(SVector{4, Int}, y), particles(b))
end

using ParticleFilters

up = BootstrapFilter(pomdp, 100)
init = initialstate(pomdp)
b = initialize_belief(up, init)

fieldnames(typeof(particle(b,1)))
@time vcat(particle(b,1).robot, particle(b,1).opponent)

function default_input_representation(b::AbstractParticleBelief)
    stack(particles(b)) do s
        convert(AbstractVector{Float32}, s)
    end
end

default_input_representation(b)

particle(b,1)
convert_s(AbstractVector, particle(b,1))


function f()
    pomdp = cpp_emu_lasertag(4)
    up = BootstrapFilter(pomdp, 100)
    init = initialstate(pomdp)
    b = initialize_belief(up, init)
    @time stack(particles(b)) do s
        vcat(s.robot, s.opponent)
    end
end
f()


LaserTagPOMDP{}(
    10.0,
    1.0,
    0.95,
    LaserTag.Floor(7,11),
    Set{LaserTag.Coord}(),
    nothing,
    false,
    LaserTag.LTDistanceCache(LaserTag.Floor(7,11), Set{LaserTag.Coord}()),
    LaserTag.DESPOTEmu(LaserTag.Floor(7,11), 2.5)
)

# @everywhere include("models/lasertag.jl")
# @everywhere using .LaserTag

# @everywhere function minBetaZero.input_representation(b::AbstractParticleBelief{<:LTState})
#     stack(y->convert(SVector{4, Int}, y), particles(b))
# end

@everywhere using LaserTag


@everywhere mean_std_layer(x) = dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)

params = minBetaZeroParameters(
    GumbelSolver_args = (;
        max_depth           = 10,
        n_particles         = 100,
        tree_queries        = 100,
        max_time            = Inf,
        k_o                 = 10.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 1_000, 
        beliefcache_size    = 1_000,
        m_acts_init         = length(actions(DiscreteLaserTagPOMDP())),
        cscale              = 1.,
        cvisit              = 50.
    ),
    t_max           = 100,
    n_episodes      = 50,
    n_iter          = 500,
    batchsize       = 128,
    lr              = 3e-4,
    lambda          = 1e-3,
    plot_training   = false,
    train_device    = gpu,
    inference_device = gpu,
    buff_cap = 50_000,
    train_intensity = 8,
    warmup_steps = 10_000,
    input_dims = (4,)
)

function f(buff_cap, train_intensity, warmup_steps)
    sum(1 - ((i-1)/i)^train_intensity for i in warmup_steps:buff_cap)
end

f(50_000, 8, 10_000)

nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size         = length(actions(DiscreteLaserTagPOMDP())),
    input_size          = (4,),
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(-100:10:100),
    p_dropout           = 0.1,
    neurons             = 64,
    hidden_layers       = 2,
    shared_net          = mean_std_layer,
    shared_out_size     = (4*2,) # must manually set... fix at a later date...        
)


results = []
for _ in 1:5
    @time net, info = betazero(params, DiscreteLaserTagPOMDP(), ActorCritic(nn_params));

    plot(
        plot(info[:steps], info[:returns]; label=false, xlabel="Steps", title="Mean Episodic Return"),
        plot(info[:episodes], info[:returns]; label=false, xlabel="Episodes");
        layout=(2,1)
    ) |> display

    push!(results, (net, info))
end

# p = plot(xlabel="Episodes", title="Mean Episodic Return - $(params.buff_cap) Buffer - $(params.n_episodes) Episodes")
# for (_, info) in results
#     plot!(p, info[:episodes], info[:returns]; label=false)
# end
# p

p = plot()

x = results[1][2][:episodes]
y = stack(x->x[2][:returns], results)
mu, bounds = ci_bounds(y)
error_plot!(p, x, mu, bounds; label="Intensity = 16 - 50k", xlabel="Episodes", ylabel="Return", title="Mean Episodic Return")

function ci_bounds(y; ci=0.95, dim=argmin(size(y)))
    n = size(y, dim)
    t = quantile(TDist(n-1), 1-(1-ci)/2)
    mu = mean(y; dims=dim)
    diff = t * std(y; dims=dim) / sqrt(n)
    bounds = mu .+ cat(-diff, diff; dims=dim)
    return mu, bounds
end

error_plot(args...; kwargs...) = error_plot!(plot(), args...; kwargs...)
error_plot!(p, y, bounds; kwargs...) = error_plot!(p, 1:length(y), y, bounds; kwargs...)
function error_plot!(p, x, y, bounds; c=1+p.n÷2, fillcolor=c, fillalpha=0.2, linealpha=0, bounds_label=false, label=false, linewidth=1.5, kwargs...)
    plot!(p, x, bounds[:,1]; fillrange = bounds[:,2], fillalpha, linealpha, fillcolor, label=bounds_label, kwargs...)
    plot!(p, x, y; label, linewidth, c, kwargs...)
end


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


