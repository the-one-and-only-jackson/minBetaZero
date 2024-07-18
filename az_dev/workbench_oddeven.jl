using POMDPs, POMDPTools, ParticleFilters
using Statistics, StatsBase, Distributions, Random, SparseArrays
using Flux, CUDA
using Plots

includet("odd_even.jl")
using .OddEven

includet("BeliefMDPs.jl")
using .BeliefMDPs

includet("AlphaZero/AlphaZero.jl")
using .AlphaZero

includet("plot_smoothing.jl")

function f_init(x::AbstractArray{T}; thresh=T(1e-3)) where {T <: AbstractFloat}
    @. 1 - inv(log(thresh)) * log(max(x, thresh))
end

mdp = ExactBeliefMDP(OddEvenPOMDP(; num_states=10, rlow=-1, rhigh=0, discount=0.95))
na = length(actions(mdp))
input_dims = size(rand(MersenneTwister(1), initialstate(mdp)))
ns = prod(input_dims)

params = AlphaZeroParams(;
    buff_cap = 1_000_000,
    warmup_steps = 50_000,
    steps_per_iter = 20_000,
    inference_batchsize = 32,
    batchsize = 1024,
    lr = 1e-4,
    lambda = 1e-3,
    n_iter = 20,
    inference_T = Float32,
    value_scale = 0.5,
    plot_training = true,
    tree_queries = 10,
    k_o = 1,
    m_acts_init = 10,
    max_steps = 100,
    segment_length = 4
)

nn_params = NetworkParameters(
    action_size         = na,
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(range(-20, 0, length=64)),
    p_dropout           = 0.0,
    neurons             = 64,
    hidden_layers       = 2,
    shared_net          = f_init,
    shared_out_size     = (10,),
    activation          = gelu
)

ac = ActorCritic(nn_params)
net, info = alphazero(params, mdp, ac);

plot_smoothing(info[:steps], info[:returns]; k=10)

info[:training][1]


function testfun(mdp)
    ret = 0.0
    s = rand(initialstate(mdp))
    for t in 0:500
        a = argmax(s)
        sp, r = @gen(:sp, :r)(mdp, s, a)
        s = sp
        ret += r * discount(mdp)^t
    end
    return ret
end

mdp = ExactBeliefMDP(OddEvenPOMDP(; num_states=10, rlow=-1, rhigh=0, discount=0.95))
y = [testfun(mdp) for _ in 1:1000]
mean(y)


x = zeros(10, 10_000)
t = 0
s = rand(initialstate(mdp))
for i in axes(x, 2)
    a = rand(1:10)
    sp, r = @gen(:sp, :r)(mdp, s, a)
    x[:, i] .= s = sp

    t += 1
    if t == 100
        t = 0
        s = rand(initialstate(mdp))
    end
end

function entropy(b)
    sum(p -> iszero(p) ? zero(p) : -p * log(p), b)
end

function testnnz(mdp; n_steps = 10_000, thresh=1e-3)
    s = rand(initialstate(mdp))
    a = 0
    for t in 0:n_steps-1
        a = rand(actions(mdp))
        sp, r = @gen(:sp, :r)(mdp, s, a)
        s = sp
    end
    entropy(s)
end

stepitr = [0; sort(vec([10^i * j for (i, j) in Iterators.product(0:3, 1:9)])); 10_000]

x = zeros(10_000, length(stepitr))

for (j, n_steps) in enumerate(stepitr)
    Threads.@threads for i in Base.axes(x, 1)
        x[i, j] = testnnz(mdp; n_steps)
    end
end

anim = @animate for (j, n_steps) in enumerate(stepitr)
    histogram(exp.(x[:, j]);
        bins=range(0.5,10.5,11), weights=fill(1/size(x,1), size(x,1)),
        xlims=(0,11), xticks=1:10,
        ylims=(0,1), label=false, xlabel="Entropy (Equivalent Uniform(n))", title="Steps: $n_steps"
    )
end

gif(anim, "entropy.gif", fps=3)
