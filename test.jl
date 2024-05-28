using minBetaZero

using ParticleFilters
using Statistics
using Flux

include("models/LightDark.jl")
using .LightDark

# minBetaZero.setup()

function minBetaZero.input_representation(b::AbstractParticleBelief{<:LightDarkState})
    out_arr = Array{Float32}(undef, (1, n_particles(b)))
    for (j, p) in enumerate(particles(b))
        out_arr[1,j] = p.y
    end
    return out_arr
end

nn_params = NetworkParameters(
    action_size=3,
    input_size=(1,),
    critic_loss = Flux.Losses.mse,
    p_dropout = 0.2,
    neurons = 64,
    shared_net = Chain(
        x->dropdims(cat(mean(x; dims=2), std(x; dims=2); dims=1); dims=2),
        Flux.Scale(2)
    ),
    shared_out_size = 2 # must manually set... fix at a later date...        
)

betazero(LightDarkPOMDP(); nn_params, n_episodes=500, t_max=100, noise_alpha=0.1)


