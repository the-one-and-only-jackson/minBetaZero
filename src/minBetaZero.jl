module minBetaZero

using Flux, CUDA
using POMDPs, POMDPModelTools, POMDPTools
using ParticleFilters
using Statistics, Distributions, Random
using Pkg
using ProgressMeter, Plots

# include("../lib/ParticleFilterTrees/src/ParticleFilterTrees.jl")
using ParticleFilterTrees

include("beliefmdp.jl")
export ParticleBeliefMDP

include("neural_network.jl")
using .NeuralNet
export NetworkParameters, ActorCritic, getloss

export betazero, minBetaZeroParameters, input_representation

function input_representation end

include("tools.jl")

@kwdef struct minBetaZeroParameters
    GumbelSolver_args = (;
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
        cscale              = 1.,
        cvisit              = 50.,
        stochastic_root     = true,
        m_acts_init
    )
    t_max       = 100
    n_episodes  = 500
    n_iter      = 20
    train_frac  = 0.8
    batchsize   = 128
    lr          = 3e-4
    lambda      = 0.0
    n_epochs    = 50
    plot_training = false
    train_device  = cpu
    inference_device = cpu
    early_stop  = 10
    n_warmup    = 500
    input_dims = (1,500)
    na = 3
    buff_cap = 10_000
    n_particles = 500
end

function betazero(params::minBetaZeroParameters, pomdp::POMDP, net)
    (; n_iter, input_dims, na, buff_cap) = params

    info = Dict(
        :steps => Int[],
        :episodes => Int[],
        :returns => Float64[],
        :returns_error => Float64[],
        :training => Dict[]
    )

    buffer = DataBuffer(input_dims, na, buff_cap)

    for itr in 1:n_iter+1
        iter_timer = time()

        @info itr <= n_iter ? "Gathering Data - Iteration: $itr" : "Gathering Data - Final Evaluation"
        returns, steps = gen_data(pomdp, net, params, buffer)
        
        episodes = length(returns)
        returns_mean = mean(returns)
        returns_error = std(returns)/sqrt(length(returns))

        push!.(
            (info[:steps], info[:episodes], info[:returns], info[:returns_error]), 
            (steps, episodes, returns_mean, returns_error)
        )

        _rm = round(returns_mean; sigdigits=1+Base.hidigit(returns_mean,10)-Base.hidigit(returns_error,10))
        _re = round(returns_error; sigdigits=1)
        @info "Mean return $_rm +/- $_re"
        @info "Gathered $steps data points over $episodes episodes"

        # This is so we get the stats after the n_iter network update
        itr == n_iter+1 && break

        train_data = (; 
            x = select_last_dim(buffer.network_input, 1:buffer.length),
            value_target = select_last_dim(buffer.value_target, 1:buffer.length),
            policy_target = select_last_dim(buffer.policy_target, 1:buffer.length),
        )

        net, train_info = train!(net, train_data, params)
        push!(info[:training], train_info)

        iter_time = ceil(Int, time()-iter_timer)
        @info "Iteration time: $iter_time seconds"
        println()
    end

    info[:steps] = cumsum(info[:steps])
    info[:episodes] = cumsum(info[:episodes])

    return net, info
end

function gen_data(pomdp, net, params::minBetaZeroParameters, buffer::DataBuffer)
    (; n_episodes, inference_device) = params
    
    worker_querries  = [Channel{Any}(1) for _ in 1:n_episodes]
    master_responses = [Channel{Any}(1) for _ in 1:n_episodes]
    data_channel = Channel{Any}(n_episodes)

    progress = Progress(n_episodes)
    steps = 0
    ret_vec = Float32[]

    @sync begin
        Threads.@spawn try
            master_function(net, inference_device, worker_querries, master_responses)
        catch e
            map(close, worker_querries)
            map(close, master_responses)
            close(data_channel)
            rethrow(e)
        end

        for (querry_channel, response_channel) in zip(worker_querries, master_responses) 
            Threads.@spawn try 
                data = work_fun(pomdp, params, querry_channel, response_channel)
                put!(data_channel, data)
                close(querry_channel)
                close(response_channel)
            catch e
                close(querry_channel)
                close(response_channel)
                close(data_channel)
                rethrow(e)
            end
        end

        Threads.@spawn begin
            while isready(data_channel) || any(isopen, worker_querries)
                (; belief, value, policy, returns) = take!(data_channel)
                set_buffer!(buffer; network_input=belief, value_target=value, policy_target=policy)
                push!(ret_vec, returns)
                steps += length(value)
                next!(progress)
            end
        end
    end

    return ret_vec, steps
end

function master_function(net, device, worker_querries, master_responses)
    master_net = net |> device

    while any(isopen, worker_querries)
        idxs = isready.(worker_querries)
        !any(idxs) && continue

        if count(idxs) == 1
            idx = findfirst(idxs)
            querries = take!(worker_querries[idx])
            querries_batched = reshape(querries, size(querries)..., 1)
            results = master_net(querries_batched |> device; logits=true) |> cpu
            put!(master_responses[idx], results)    
        else
            querries = take!.(worker_querries[idxs])
            querries_batched = querries |> stack
            results = master_net(querries_batched |> device; logits=true) |> cpu
            dims = size(first(results),ndims(first(results)))
            split_results = [(; value=results.value[:,i], policy=results.policy[:,i]) for i in 1:dims]
            put!.(master_responses[idxs], split_results)
        end
    end

    return nothing
end

function work_fun(pomdp, params, querry_channel, response_channel)
    (; t_max, GumbelSolver_args, n_particles) = params

    function getpolicyvalue(b)
        b_rep = input_representation(b)
        put!(querry_channel, b_rep)
        out = take!(response_channel)
        return (; value=out.value[], policy=vec(out.policy))
    end

    planner = solve(GumbelSolver(; getpolicyvalue, GumbelSolver_args...), pomdp)

    up = BootstrapFilter(pomdp, n_particles)
    b = initialize_belief(up, initialstate(pomdp))
    s = rand(initialstate(pomdp))

    b_vec = typeof(b)[]
    aid_vec = Int[]
    r_vec = Float32[]
    for _ in 1:t_max
        a = action(planner, b)
        aid = actionindex(pomdp,a)
        s, r, o = @gen(:sp,:r,:o)(pomdp, s, a)
        push!.((b_vec,aid_vec,r_vec), (b,aid,r))
        b = POMDPs.update(up, b, a, o)
        if isterminal(pomdp, s)
            break
        end
    end

    gamma = discount(pomdp)
    for i in length(r_vec)-1:-1:1
        r_vec[i] += gamma * r_vec[i+1]
    end

    data = (; 
        belief = stack(input_representation, b_vec), 
        value = reshape(r_vec, 1, :), 
        policy = Flux.onehotbatch(aid_vec, 1:length(actions(pomdp))), 
        returns = r_vec[1]
    )

    return data
end

function train!(net, data, params::minBetaZeroParameters)
    (; train_frac, batchsize, lr, lambda, n_epochs, train_device, early_stop, n_warmup, plot_training) = params

    Etrain = mean(-sum(x->iszero(x) ? x : x*log(x), data.policy_target; dims=1))
    varVtrain = var(data.value_target)

    train_batchsize = min(batchsize, Flux.numobs(data))
    train_data = Flux.DataLoader(data; batchsize=train_batchsize, shuffle=true, partial=false) |> train_device

    @info "Training on $(Flux.numobs(data)) samples"

    net = train_device(net)
    opt = Flux.setup(Flux.Optimiser(Flux.Adam(lr), WeightDecay(lr*lambda)), net)

    info = Dict(:policy_loss=>Float32[], :policy_KL=>Float32[], :value_loss=>Float32[], :value_FVU=>Float32[])
        
    Flux.trainmode!(net)
    @showprogress for (; x, value_target, policy_target) in train_data
        grads = Flux.gradient(net) do net
            losses = getloss(net, x; value_target, policy_target)

            Flux.Zygote.ignore_derivatives() do 
                push!(info[:policy_loss], losses.policy_loss          )
                push!(info[:policy_KL]  , losses.policy_loss - Etrain )
                push!(info[:value_loss] , losses.value_loss           )
                push!(info[:value_FVU]  , losses.value_mse / varVtrain)
                return nothing
            end

            return losses.value_loss + losses.policy_loss
        end
        Flux.update!(opt, net, grads[1])
    end
    Flux.testmode!(net)

    if plot_training
        plotargs = (; label=false)
        plot(
            plot(info[:value_loss]; ylabel="Value Loss", plotargs...),
            plot(info[:policy_loss]; ylabel="Policy Loss", plotargs...),
            plot(info[:value_FVU]; ylabel="FVU", plotargs...),
            plot(info[:policy_KL]; ylabel="Policy KL", plotargs...)
            ; 
            layout=(2,2), 
            size=(900,600)
        ) |> display
    end

    return cpu(net), info
end




end # module minBetaZero




#= Notes

For all of the data in the buffer, update

trains on the data "reuse" times

gather data
states = 0
for trajectory in generator()
    add to buffer
    states += new_states
    if states > buff_cap / reuse
        break
    end
end




=#