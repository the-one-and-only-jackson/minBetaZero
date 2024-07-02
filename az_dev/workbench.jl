using POMDPs, POMDPTools, ParticleFilters
using Statistics, StatsBase, Distributions, Random, SparseArrays
using Flux, CUDA
using Plots

includet("lasertag.jl")
using .LaserTag

includet("BeliefMDPs.jl")
using .BeliefMDPs

includet("AlphaZero/AlphaZero.jl")
using .AlphaZero

includet("plot_smoothing.jl")

function f_init(x::T) where {T <: AbstractFloat}
    thresh = T(1f-5)
    scale  = T(20)
    y = one(T) - inv(log(thresh)) * log(max(x, thresh))
    y * scale
end

function f_init(x::AbstractArray{T}; thresh=T(1e-5), scale=T(20)) where {T <: AbstractFloat}
    coeff  = scale / log(thresh)
    @. scale - coeff * log(max(x, thresh))
end

mdp = ExactBeliefMDP(LaserTagPOMDP())
na = length(actions(mdp))
input_dims = size(rand(MersenneTwister(1), initialstate(mdp)))
ns = prod(input_dims)

params = AlphaZeroParams(;
    n_iter = 1000,
    buff_cap = 500_000,
    warmup_steps = 50_000,
    steps_per_iter = 10_000,
    inference_batchsize = 16,
    n_agents = 32,
    batchsize = 1024,
    lr = 1e-3,
    lambda = 1e-2,
    inference_T = Float32,
    value_scale = 0.5,
    plot_training = true,
    tree_queries = 10,
    k_o = 5,
    max_steps = 1000,
    segment_length = typemax(Int)
)

nn_params = NetworkParameters(
    action_size         = na,
    critic_loss         = Flux.Losses.logitcrossentropy,
    critic_categories   = collect(range(-100, 100, length=256)),
    p_dropout           = 0.0,
    neurons             = 512,
    hidden_layers       = 1,
    activation          = gelu,
    shared_net          = Chain(
        f_init,
        x -> reshape(x, 70, 70, 1, :),
        Conv((3,3), 1=>32, gelu), # (68, 68, 32, N)
        MaxPool((2,2)), # (34, 34, 32, N)
        Conv((3,3), 32=>64, gelu), # (32, 32, 64, N)
        MaxPool((2,2)), # (16, 16, 64, N)
        Conv((3,3), 64=>128, gelu; pad=SamePad()), # (16, 16, 128, N)
        MaxPool((2,2)), # (8, 8, 128, N)
        Conv((2,2), 128=>256, gelu; pad=SamePad()), # (8, 8, 256, N)
        MaxPool((2,2)), # (4, 4, 256, N)
        Conv((2,2), 256=>512, gelu; pad=SamePad()), # (4, 4, 512, N)
        MaxPool((2,2)), # (2, 2, 512, N)
        Conv((2,2), 512=>1024, gelu), # (1, 1, 1024, N)
        Flux.flatten, # (1024, N)
        Dense(1024=>512, gelu)
    ),
    shared_out_size = (512,)
)

AlphaZero.NeuralNet.DEBUG[] = false

info = Dict{Symbol, Any}()
net = ActorCritic(nn_params)
net, info = alphazero(params, mdp, net, info)

plot_smoothing(info[:steps], info[:returns]; k=10)

p = Flux.params(info[:ac])
extrema.(p)


#

# collect a bunch of Data

function collect_ep(mdp, ri, cs, cv)
    rng = MersenneTwister(ri)

    s = Vector{Float32}[]
    v = Float32[]

    sp = rand(initialstate(mdp))

    for _ in 1:1000
        push!(s, sp)
        a = rand(rng, actions(mdp))
        (sp, r) = @gen(:sp, :r)(mdp, sp, a, rng)
        push!(v, r)
        isterminal(mdp, sp) && break
    end

    gamma = Float32(discount(mdp))

    for i in Iterators.reverse(eachindex(v))
        i == lastindex(v) && continue
        v[i] += gamma * v[i+1]
    end

    put!(cs, s)
    put!(cv, v)

    return nothing
end

N = 1_000
cin = Channel{Int}(N)
cs  = Channel{Vector{Vector{Float32}}}(N)
cv  = Channel{Vector{Float32}}(N)

foreach(i -> put!(cin, i), 1:N)
close(cin)

Threads.foreach(i -> collect_ep(mdp, i, cs, cv), cin)

ns = sum(length, cs.data)
S = zeros(Float32, 4900, ns);
V = zeros(Float32, ns);

i = 1
for s_vec in cs.data, s in s_vec
    S[:, i] .= s
    i += 1
end

i = 1
for v_vec in cv.data, v in v_vec
    V[i] = v
    i += 1
end

V .= (V .- mean(V)) ./ std(V)


f_init(x; thresh=1f-5, scale=20) = x < thresh ? zero(x) : scale * (1 - log10(x)/log10(thresh))

function conv_layers(; n1 = 128)
    Chain(
        x -> reshape(x, 10, 7, 70, :),
        Conv((3,2), 70=>n1, gelu; pad=(0, 1)), # (8, 8, n1, N)
        Flux.SkipConnection(
            Chain(
                x -> reshape(x, 8*8*n1, :),
                RMSNorm(8*8*n1),
                x -> reshape(x, 8, 8, n1, :),
                Conv((2,2), n1=>n1, gelu; pad=SamePad())
            ),
            +
        ),
        MaxPool((2,2)), # (4, 4, n1, N)
        Conv((2,2), n1=>n1, gelu), # (3, 3, n1, N)
        Conv((2,2), n1=>n1, gelu), # (2, 2, n1, N)
        Flux.flatten, # (4*n1, N)
        RMSNorm(4*n1)
    )
end

# c = Chain(
#     x -> f_init.(x),
#     conv_layers(; n1 = 256),
#     Dense(4*256=>512, tanh),
#     Dense(512=>1)
# ) |> gpu

# c = Chain(
#     x -> f_init.(x),
#     Dense(4900=>1024, tanh),
#     Dense(1024=>256, tanh),
#     Dense(256=>1)
# ) |> gpu

c = Chain(
    x -> f_init.(x),
    x -> reshape(x, 70, 70, 1, :),
    Conv((3,3), 1=>32, gelu), # (68, 68, 32, N)
    MaxPool((2,2)), # (34, 34, 32, N)
    Conv((3,3), 32=>64, gelu), # (32, 32, 64, N)
    MaxPool((2,2)), # (16, 16, 64, N)
    Conv((3,3), 64=>128, gelu; pad=SamePad()), # (16, 16, 128, N)
    MaxPool((2,2)), # (8, 8, 128, N)
    Conv((2,2), 128=>256, gelu; pad=SamePad()), # (8, 8, 256, N)
    MaxPool((2,2)), # (4, 4, 256, N)
    Conv((2,2), 256=>512, gelu; pad=SamePad()), # (4, 4, 512, N)
    MaxPool((2,2)), # (2, 2, 512, N)
    Conv((2,2), 512=>1024, gelu), # (1, 1, 1024, N)
    Flux.flatten, # (1024, N)
    Dense(1024=>1)
) |> gpu

lambda = 1e-4
eta = 1e-2
optimiser = Flux.Optimisers.OptimiserChain(
    Flux.Optimisers.ClipNorm(1),
    Flux.Optimisers.ClipGrad(1),
    Flux.Optimisers.AdamW(; eta, lambda = lambda * eta)
)

opt = Flux.setup(optimiser, c)

bs = 64

s_cpu = zeros(Float32, 4900, bs)
s_gpu = CUDA.zeros(Float32, 4900, bs)
v_cpu = zeros(Float32, 1, bs)
v_gpu = CUDA.zeros(Float32, 1, bs)

l = zeros(Float32, 100_000)
fvu = zeros(Float32, length(l))
p = Progress(length(l))

@showprogress for i in eachindex(l)
    idxs = rand(1:ns, bs)
    copyto!(s_cpu, @view S[:, idxs])
    copyto!(v_cpu, @view V[idxs])
    copyto!(s_gpu, s_cpu)
    copyto!(v_gpu, v_cpu)

    l[i], grads = Flux.withgradient(c) do net
        L = Flux.mse(net(s_gpu), v_gpu)
        Flux.Zygote.ignore_derivatives() do
            fvu[i] = L / var(v_gpu)
            return nothing
        end
        return L
    end

    Flux.update!(opt, c, grads[1])

    next!(p; showvalues = [(:loss, l[i]), (:fvu, fvu[i])])
end

plot(fvu; ylims=(0, 2))

plot_smoothing(l; k=10, ylims=(0.8,1.2))
