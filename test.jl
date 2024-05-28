using Distributed

addprocs(10)

@everywhere begin
    using minBetaZero
    using ParticleFilters
    using Statistics
    using Flux
    # include("models/LightDark.jl")
    using POMDPTools, QuickPOMDPs, POMDPs, Distributions
end

# @everywhere begin
#     using .LightDark
# end

# minBetaZero.setup()

@everywhere begin
    const R = 60
    const LIGHT_LOC = 10
    LightDarkPOMDP() = QuickPOMDP(
        states = -R:R+1, # r+1 is a terminal state
        stateindex = s -> s + R + 1,
        actions = [-10, -1, 0, 1, 10],
        discount = 0.95,
        isterminal = s::Int -> s==R::Int+1,
        obstype = Float64,

        transition = function (s::Int, a::Int)
            if a == 0
                return Deterministic{Int}(R::Int+1)
            else
                return Deterministic{Int}(clamp(s+a, -R::Int, R::Int))
            end
        end,

        observation = (s, a, sp) -> Normal(sp, abs(sp - LIGHT_LOC::Int) + 1e-3),

        reward = function (s, a)
            if iszero(a)
                return iszero(s) ? 100.0 : -100.0
            else
                return -1.0
            end
        end,

        initialstate = POMDPTools.Uniform(div(-R::Int,2):div(R::Int,2))
    )

    function minBetaZero.input_representation(b::PFTBelief{<:Number})
        reshape(convert(Vector{Float32}, b.particles), 1, n_particles(b))
    end
end





# @everywhere function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
#     out_arr = Array{Float32}(undef, (1, n_particles(b)))
#     for (j, p) in enumerate(particles(b))
#         out_arr[1,j] = p.y
#     end
#     return out_arr
# end

nn_params = NetworkParameters(
    action_size=5,
    input_size=(1,),
    critic_loss = Flux.Losses.logitcrossentropy,
    critic_categories = collect(-100:10:100),
    p_dropout = 0.2,
    neurons = 64,
    shared_net = Chain(
        x->dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)
    ),
    shared_out_size = 2 # must manually set... fix at a later date...        
)

betazero(LightDarkPOMDP(); nn_params, n_episodes=500, t_max=100, noise_alpha=0.)





pomdp = LightDarkPOMDP()
net = ActorCritic(nn_params)
data = minBetaZero.gen_data(pomdp, net; t_max=100, n_episodes=500)
data.value
data.belief

net(data.belief).policy
mean(net(data.belief).value)
std(net(data.belief).value)

7795 / 500

nw = NetworkWrapper(; net)
solver = PFTDPWSolver(
    value_estimator     = nw,
    policy_estimator    = PUCT(; net=nw, c=1.0),
    max_depth           = 10,
    n_particles         = 100,
    tree_queries        = 100,
    max_time            = Inf,
    k_a                 = 2.0,
    alpha_a             = 0.25,
    k_o                 = 2.0,
    alpha_o             = 0.1,
    check_repeat_obs    = false,
    resample            = true,
    treecache_size      = 10_000, 
    beliefcache_size    = 10_000, 
)
planner = minBetaZero.solve(solver, pomdp)

bmdp = ParticleBeliefMDP(pomdp)

b = rand(initialstate(bmdp))

b_vec = typeof(b)[]
p_vec = Vector{Float32}[]
r_vec = Float32[]
term_flag = false

for _ in 1:100
    empty!(nw)
    _, a_info = action_info(planner, b)
    p = minBetaZero.calculate_targetdist(bmdp.pomdp, a_info.tree)
    a_idx = minBetaZero.sample_cat(p) # bmdp uses action indexes
    bp, r = @gen(:sp,:r)(bmdp, b, a_idx)
    push!.((b_vec,p_vec,r_vec), (b,p,r))
    b = bp
    if minBetaZero.ParticleFilterTrees.isterminalbelief(b)
        term_flag = true
        break
    end
end

y = stack([[p.y for p in particles(b)] for b in b_vec])
scatter(y'; labels=false, c=1)

data = minBetaZero.gen_data(pomdp, net; t_max=100, n_episodes=500)

noise = rand(Dirichlet(size(data.policy,1), Float32(0.1)), size(data.policy,2))
alpha = Float32(0.25)
noisy_policy = (1-alpha)*data.policy + alpha*noise


batchsize = 1024
_data = (; value_target = data.value, policy_target = data.policy, x=dropdims(cat(mean(data.belief; dims=2), std(data.belief; dims=2); dims=1); dims=2))
split_data = Flux.splitobs(_data, at=0.7)
train_data = Flux.DataLoader(split_data[1]; batchsize=min(batchsize, Flux.numobs(split_data[1])), shuffle=true, partial=false)
valid_data = Flux.DataLoader(split_data[2]; batchsize=min(batchsize, Flux.numobs(split_data[2])), shuffle=true, partial=false)

Etrain = mean(-sum(x->iszero(x) ? x : x*log(x), train_data.data.policy_target; dims=1))
Evalid = mean(-sum(x->iszero(x) ? x : x*log(x), valid_data.data.policy_target; dims=1))
varVtrain = var(train_data.data.value_target)
varVvalid = var(valid_data.data.value_target)

net = ActorCritic(NetworkParameters(
    action_size=3,
    input_size=(2,),
    critic_loss = Flux.Losses.logitcrossentropy,
    critic_categories = -100:10:100,
    p_dropout = 0.0,
    neurons = 128,
))

opt = Flux.setup(Flux.Optimiser(Flux.Adam(1e-3), WeightDecay(0.0)), net)

info = Dict(
    :value_train_loss  => Float32[],
    :value_valid_loss  => Float32[],
    :policy_train_loss => Float32[],
    :policy_valid_loss => Float32[],
    :train_R           => Float32[],
    :valid_R           => Float32[],
    :train_KL          => Float32[],
    :valid_KL          => Float32[]
)

n_epochs = 100
@showprogress for _ in 1:n_epochs
    Flux.trainmode!(net)
    for (; x, value_target, policy_target) in train_data    
        grads = Flux.gradient(net) do net
            losses = getloss(net, x; value_target, policy_target)
            Flux.Zygote.ignore_derivatives() do 
                push!(info[:policy_train_loss], losses.policy_loss)
                push!(info[:train_KL], losses.policy_loss - Etrain)
                push!(info[:value_train_loss], losses.value_loss)
            end
            losses.value_loss + losses.policy_loss
        end
        Flux.update!(opt, net, grads[1])

        push!(info[:train_R], Flux.mse(net(x).value, value_target)/varVtrain)
    end

    Flux.testmode!(net)
    for (; x, value_target, policy_target) in valid_data    
        losses = getloss(net, x; value_target, policy_target)
        push!(info[:policy_valid_loss], losses.policy_loss)
        push!(info[:valid_KL], losses.policy_loss - Evalid)
        push!(info[:value_valid_loss], losses.value_loss)
        push!(info[:valid_R], Flux.mse(net(x).value, value_target)/varVvalid)
    end
end

for (k,v) in info
    info[k] = dropdims(mean(reshape(v,:,n_epochs); dims=1); dims=1)
end

pv = plot(info[:train_R]; c=1, label="train", ylabel="FVU", title="Loss")
plot!(pv, info[:valid_R]; c=2, label="valid")
pp = plot(info[:train_KL]; c=1, label="train", ylabel="Policy KL")
plot!(pp, info[:valid_KL]; c=2, label="valid", xlabel="Epochs")
plot(pv, pp; layout=(2,1)) |> display




actor = ActorCritic(NetworkParameters(
    action_size=3,
    input_size=(2,),
    critic_loss = Flux.Losses.logitcrossentropy,
    critic_categories = -100:10:100,
    p_dropout = 0.2,
    neurons = 64,
)).actor

opt = Flux.setup(Flux.Optimiser(Flux.Adam(1e-3), WeightDecay(0.0)), actor)

info = Dict(
    :value_train_loss  => Float32[],
    :value_valid_loss  => Float32[],
    :policy_train_loss => Float32[],
    :policy_valid_loss => Float32[],
    :train_R           => Float32[],
    :valid_R           => Float32[],
    :train_KL          => Float32[],
    :valid_KL          => Float32[]
)

n_epochs = 100
@showprogress for _ in 1:n_epochs
    Flux.trainmode!(actor)
    for (; x, value_target, policy_target) in train_data    
        grads = Flux.gradient(actor) do net
            losses = getloss(net, x; policy_target)
            Flux.Zygote.ignore_derivatives() do 
                push!(info[:policy_train_loss], losses)
                push!(info[:train_KL], losses - Etrain)
            end
            losses
        end
        Flux.update!(opt, actor, grads[1])
    end

    Flux.testmode!(actor)
    for (; x, value_target, policy_target) in valid_data    
        losses = getloss(actor, x; policy_target)
        push!(info[:policy_valid_loss], losses)
        push!(info[:valid_KL], losses - Evalid)
    end
end

for (k,v) in info
    info[k] = dropdims(mean(reshape(v,:,n_epochs); dims=1); dims=1)
end
pp = plot(info[:train_KL]; c=1, label="train", ylabel="Policy KL")
plot!(pp, info[:valid_KL]; c=2, label="valid", xlabel="Epochs")




#=
TODO

put mse in getloss
=#





