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






