using minBetaZero

using POMDPs
using POMDPTools
using ParticleFilters
using Flux
using Statistics
using Plots
using ProgressMeter, ParticleFilterTrees

include("models/LightDark.jl")
using .LightDark

function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
    rep = Float32[p.y for p in particles(b)]
    reshape(rep, 1, :)
end

mean_std_layer(x) = dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)


for cscale in [0.1, 1., 10., 100.], cvisit in [1., 3., 10., 30.]
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
            treecache_size      = 1_000, 
            beliefcache_size    = 1_000,
            m_acts_init         = 3,
            cscale = cscale,
            cvisit = cvisit
        ),
        t_max = 50,
        n_episodes = 500,
        n_iter = 20,
        train_frac = 0.7,
        batchsize = 512,
        lr = 3e-4,
        lambda = 0.0,
        n_epochs = 500,
        plot_training = false,
        train_dev = gpu,
        early_stop = 50,
        n_warmup = 500
    )

    nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
        action_size=3,
        input_size=(1,),
        critic_loss = Flux.Losses.logitcrossentropy,
        critic_categories = collect(-100:10:100),
        p_dropout = 0.1,
        neurons = 64,
        hidden_layers = 2,
        shared_net = mean_std_layer,
        shared_out_size = (2,) # must manually set... fix at a later date...        
    )

    @time net, info = betazero(params, LightDarkPOMDP(), ActorCritic(nn_params));

    plot(
        plot(info[:steps], info[:returns]; label=false, xlabel="Steps", title="Mean Episodic Return"),
        plot(info[:episodes], info[:returns]; label=false, xlabel="Episodes");
        layout=(2,1)
    )

    savefig("scale_$(cscale)_visit_($cvisit).png")
end

d = eval_net_only(LightDarkPOMDP(), net; n_episodes=300, n_workers=20)
mean(d)
std(d)/sqrt(length(d))

length(d)

function eval_net_only(pomdp, net; t_max=100, n_episodes=256, n_workers=64)
    progress = Progress(n_episodes*n_workers)
    channel = Channel{Int}(1)

    total_steps = 0
    async_loop = Threads.@spawn while true
        step = take!(channel)
        if step == 0
            finish!(progress)
            break
        elseif total_steps < progress.n # i dont know how to deal with progressmeter correctly
            total_steps += step
            if total_steps <= progress.n
                next!(progress; step)
            else
                next!(progress; step=progress.n-(total_steps-step))
            end
        end
    end

    worker_querries = [Channel{Any}(1) for _ in 1:n_workers]
    master_responses = [Channel{Any}(1) for _ in 1:n_workers]

    function work_fun(i)
        worker_net = function(x; kwargs...)
            put!(worker_querries[i], x)
            take!(master_responses[i])
        end

        up = BootstrapFilter(pomdp, 10_000)

        r_vec = zeros(n_episodes)
        for i in eachindex(r_vec)
            b = initialize_belief(up, initialstate(pomdp))
            s = rand(initialstate(pomdp))
            gamma = discount(pomdp)
            for t in 0:t_max-1
                ai = worker_net(input_representation(b)).policy |> argmax
                a = actions(pomdp)[ai]
                s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)
                r_vec[i] += r * gamma^t
                b = update(up, b, a, o)
                if isterminal(pomdp, s)
                    break
                end
            end
            put!(channel, 1)
        end
        return r_vec
    end

    master_net = net |> gpu

    futures = [Threads.@spawn work_fun(i) for i in 1:n_workers]

    while !all(istaskdone, futures)
        idxs = isready.(worker_querries)
        iszero(count(idxs)) && continue

        if count(idxs) == 1
            idx = findfirst(idxs)
            querries = take!(worker_querries[idx])
            querries_batched = reshape(querries, size(querries)..., 1) |> gpu
            results = master_net(querries_batched; logits=true) |> cpu
            put!(master_responses[idx], results)    
        else
            querries = take!.(worker_querries[idxs])
            querries_batched = querries |> stack |> gpu
            results = master_net(querries_batched; logits=true) |> cpu
            dims = size(first(results),ndims(first(results)))
            split_results = [(; value=results.value[:,i], policy=results.policy[:,i]) for i in 1:dims]
            put!.(master_responses[idxs], split_results)
        end
    end

    data = reduce(vcat, fetch.(futures))

    put!(channel, 0)
    wait(async_loop)

    return data
end



















using Distributed
addprocs(18)
@everywhere using Flux, ParticleFilters, POMDPs, POMDPTools, minBetaZero, Statistics
@everywhere using ParticleFilterTrees, Distributions

@everywhere include("models/LightDark.jl")
@everywhere using .LightDark

@everywhere function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
    rep = Float32[p.y for p in particles(b)]
    reshape(rep, 1, :)
end

@everywhere mean_std_layer(x) = dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)


results = pmap(1:nworkers()) do i
    pomdp = LightDarkPOMDP()
    up = BootstrapFilter(pomdp, 10000)

    planner = solve(
        GumbelSolver(;
            getpolicyvalue = function(x)
                out = net(input_representation(x); logits=true)
                return (; value=out.value[], policy=vec(out.policy))
            end,
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
            m_acts_init         = 3,
            stochastic_root     = false
        ),
        pomdp
    )
    
    ret_vec = zeros(50)
    for j in eachindex(ret_vec)
        b = initialize_belief(up, initialstate(pomdp))
        s = rand(initialstate(pomdp))
        gamma = discount(pomdp)
        for t in 0:49
            a, a_info = action_info(planner, b)
            s, r, o = @gen(:sp, :r, :o)(pomdp, s, a)
            ret_vec[j] += r * gamma^t
            b = update(up, b, a, o)
            isterminal(pomdp, s) && break
        end
    end
    return ret_vec
end

mcts_data = vcat(results...)
mean(mcts_data)
std(mcts_data)/sqrt(length(mcts_data))
sort(mcts_data)[floor(Int, length(mcts_data)*0.025)]
sort(mcts_data)[floor(Int, length(mcts_data)*0.975)]
# 15.185 +- 0.095

mean(mcts_data[mcts_data .> 0])
mean(mcts_data[mcts_data .< 0])



results = pmap(1:nworkers()) do _
    pomdp = LightDarkPOMDP()
    up = BootstrapFilter(pomdp, 10000)
    
    ret_vec = zeros(500)
    for j in eachindex(ret_vec)
        b = initialize_belief(up, initialstate(pomdp))
        s = rand(initialstate(pomdp))
        gamma = discount(pomdp)
        for t in 0:49
            b_est = ParticleCollection(rand(b, 100))
            b_enc = input_representation(b_est)
            a_idx = argmax(net(b_enc).policy)

            a = actions(pomdp)[a_idx]
            s, r, o = @gen(:sp, :r, :o)(pomdp, s, a)
            ret_vec[j] += r * gamma^t
            b = update(up, b, a, o)
            isterminal(pomdp, s) && break
        end
    end
    return ret_vec
end

net_data = vcat(results...)
mean(net_data)
std(net_data)/sqrt(length(net_data))
sort(net_data)[floor(Int, length(net_data)*0.025)]
sort(net_data)[floor(Int, length(net_data)*0.975)]
# 15.248 +- 0.093

mean(net_data[net_data .> 0])
mean(net_data[net_data .< 0])


results = pmap(1:nworkers()) do _
    pomdp = LightDarkPOMDP()
    up = BootstrapFilter(pomdp, 10_000)
    
    ret_vec = zeros(1000)
    for j in eachindex(ret_vec)
        b = initialize_belief(up, initialstate(pomdp))
        s = rand(initialstate(pomdp))
        gamma = discount(pomdp)
        for t in 0:99
            b_enc = input_representation(b)
            a_idx = argmax(net(b_enc; logits=true).policy + rand(Gumbel(),3))

            a = actions(pomdp)[a_idx]
            s, r, o = @gen(:sp, :r, :o)(pomdp, s, a)
            ret_vec[j] += r * gamma^t
            b = update(up, b, a, o)
            isterminal(pomdp, s) && break
        end
    end
    return ret_vec
end

net_data_2 = vcat(results...)
mean(net_data_2)
std(net_data_2)/sqrt(length(net_data_2))
sort(net_data_2)[floor(Int, length(net_data_2)*0.025)]
sort(net_data_2)[floor(Int, length(net_data_2)*0.975)]
# 14.10 +- 0.12





n_max = 30
bins=range(-n_max-0.5,n_max+0.5, length=Int(n_max*2+2))
ylims = (0, 2000)
transform(x) = log.(abs.(x)/100)/log(0.9) .* sign.(x)
histogram(transform(mcts_data); bins, ylims, label=false, title="MCTS")
histogram(transform(net_data); bins, ylims, label=false, title="Net")
histogram(transform(net_data_2); bins, ylims, label=false, title="Net Stochastic")



# TESTING


pomdp = LightDarkPOMDP()
up = BootstrapFilter(pomdp, 10000)

planner = solve(
    GumbelSolver(;
        getpolicyvalue = function(x)
            out = net(input_representation(x); logits=true)
            return (; value=out.value[], policy=vec(out.policy))
        end,
        max_depth           = 10,
        n_particles         = 100,
        tree_queries        = 10000,
        max_time            = Inf,
        k_o                 = 4.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 100_000, 
        beliefcache_size    = 100_000,
        m_acts_init         = 3,
        stochastic_root     = false
    ),
    pomdp
)



b = initialize_belief(up, initialstate(pomdp))
s = rand(initialstate(pomdp))
gamma = discount(pomdp)

a, a_info = action_info(planner, b)
check_depth(a_info.tree, 1)

tree = a_info.tree
tree.b_children[1]


function check_depth(tree, b_idx)
    c = 0
    for (_, ba_idx) in tree.b_children[b_idx]
        d = 0
        for bp_idx in tree.ba_children[ba_idx]
            d = max(d, check_depth(tree, bp_idx))
        end
        c = max(c, d)
    end
    return 1 + c
end


a_info[:Q_root]
a_info[:N_root]




q = [a_info[:Q_root][i] for i in [-1,0,1]]
logits = a_info.tree.b_P[1]
softmax(logits + 38*q)

bins = range(-5, 15, 50)
histogram([p.y for p in particles(b)]; weights=fill(1/n_particles(b), n_particles(b)), bins)
histogram!([p.y for p in particles(a_info.tree.b[1])]; bins, weights=fill(1/n_particles(a_info.tree.b[1]), n_particles(a_info.tree.b[1])))

net(input_representation(b)).policy
net(input_representation(a_info.tree.b[1])).policy
a_info.tree.b_P[1]


s, r, o = @gen(:sp, :r, :o)(pomdp, s, a)
b = update(up, b, a, o)
isterminal(pomdp, s)

net(input_representation(b)).policy

histogram(vec(input_representation(b)))












