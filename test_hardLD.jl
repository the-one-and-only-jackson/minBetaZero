using minBetaZero
using Flux, CUDA
using ParticleFilters
using Statistics
using Plots

include("models/LightDark.jl")
using .LightDark


function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
    rep = Float32[p.y for p in particles(b)]
    reshape(rep, 1, :)
end

struct CGF{T}
    weight::T
end
Flux.@layer CGF
CGF((dx,nv)::Pair{<:Integer,<:Integer}) = CGF(randn(Float32, dx, nv))
(cgf::CGF)(x) = dropdims(logsumexp(batched_mul(cgf.weight', x) ./ Float32(sqrt(size(x,1))); dims=2); dims=2) .- Float32(log(size(x,2)))

mean_std_layer(x) = dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2)

include("transformer.jl")

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
    )),
    x->(selectdim(x,2,1), selectdim(x,2,2))
)


params = minBetaZeroParameters(
    PFTDPWSolver_args = (;
        max_depth           = 10,
        n_particles         = 100,
        tree_queries        = 100,
        max_time            = Inf,
        k_a                 = 5.,
        alpha_a             = 0.,
        enable_action_pw    = false,
        k_o                 = 24.,
        alpha_o             = 0.,
        check_repeat_obs    = true,
        resample            = true,
        treecache_size      = 1_000, 
        beliefcache_size    = 1_000,
    ),
    t_max = 100,
    n_episodes = 512,
    n_workers = 512,
    n_steps = 4096,
    use_episodes = true, # gather data for at least n_episodes or at least n_steps
    c_puct = 100.0,
    n_iter = 10,
    noise_alpha = 0.25,
    noise_param = 0.1,
    train_frac = 0.8,
    batchsize = 128,
    lr = 3e-4,
    lambda = 0.0,
    n_epochs = 50,
    plot_training = true,
    train_dev = gpu,
    early_stop = 10,
    n_warmup = 500
)

nn_params = NetworkParameters( # These are POMDP specific! not general parameters - must input dimensions
    action_size=3,
    input_size=(1,),
    critic_loss = Flux.Losses.logitcrossentropy,
    critic_categories = collect(-100:10:100),
    p_dropout = 0.1,
    neurons = 64,
    hidden_layers = 1,
    shared_net = lightdark_st(),
    shared_out_size = (64,2) # must manually set... fix at a later date...        
)


@time net, info = betazero(params, LightDarkPOMDP(), ActorCritic(nn_params));


plot(
    plot(info[:steps], info[:returns]; label=false, xlabel="Steps", title="Mean Episodic Return"),
    plot(info[:episodes], info[:returns]; label=false, xlabel="Episodes");
    layout=(2,1)
)


scatter(1:length(net.shared.net.weight), net.shared.net.weight)


