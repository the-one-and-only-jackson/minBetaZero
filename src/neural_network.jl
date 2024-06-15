module NeuralNet

using Flux, CUDA, SparseArrays

export NetworkParameters, ActorCritic, getloss, CGF

include("cgf.jl")

@kwdef mutable struct NetworkParameters
    # architecture
    action_size::Int                                 # [REQUIRED] Number of actions in the action space
    input_size::Tuple{Vararg{Int}}                   # Input belief size
    activation::Function = gelu                      # Activation function
    neurons::Int = 64                             # Number of connections in fully connected layers (for CNN, refers to fully connected "head" layers)
    hidden_layers::Int = 2
    p_dropout::Float64 = 0                         # Probability of dropout
    shared_net::Any = identity                       # shared network 
    shared_out_size::Tuple{Vararg{Int}} = input_size
    critic_categories::Vector = []
    actor_loss::Function = Flux.Losses.logitcrossentropy
    critic_loss::Function = Flux.Losses.mse
end

"""
    ActorCritic
"""
struct ActorCritic{S,A,C}
    shared::S
    actor::A
    critic::C
end
Flux.@layer :expand ActorCritic

function ActorCritic(nn_params::NetworkParameters)
    @assert length(nn_params.shared_out_size) <= 2
    @assert !(length(nn_params.shared_out_size) == 2 && nn_params.shared_out_size[2] != 2)

    ActorCritic(
        SharedInput(nn_params), 
        DiscreteActor(nn_params), 
        Critic(nn_params)
    )
end

function (ac::ActorCritic)(x; logits=false)
    encoded_input = ac.shared(x)
    if encoded_input isa Tuple
        value = ac.critic(encoded_input[1])
        policy = ac.actor(encoded_input[2]; logits)
    else
        value = ac.critic(encoded_input)
        policy = ac.actor(encoded_input; logits)
    end
    return (; value, policy)
end

function getloss(ac::ActorCritic, input; value_target, policy_target)
    encoded_input = ac.shared(input)
    if encoded_input isa Tuple
        value_loss, value_mse = getloss(ac.critic, encoded_input[1]; value_target)
        policy_loss = getloss(ac.actor, encoded_input[2]; policy_target)
    else
        value_loss, value_mse = getloss(ac.critic, encoded_input; value_target)
        policy_loss = getloss(ac.actor, encoded_input; policy_target)
    end
    return (; value_loss, value_mse, policy_loss)
end

"""
    SharedInput
"""
struct SharedInput{S<:Flux.Scale, N}
    scale::S # frozen parameter input scaling
    net::N
end
Flux.@layer :expand SharedInput trainable=(net)

function SharedInput(nn_params::NetworkParameters)
    SharedInput(Flux.Scale(nn_params.input_size[1]), nn_params.shared_net)
end

function (shared::SharedInput)(x)
    scaled_x = shared.scale(x)
    shared_out = shared.net(scaled_x)
    # @assert all(isfinite, scaled_x)
    # @assert all(isfinite, shared_out) "$x"
    return shared_out
end

"""
    DiscreteActor
"""
struct DiscreteActor{NET,L}
    net::NET
    loss::L
end
Flux.@layer :expand DiscreteActor

function DiscreteActor(nn_params::NetworkParameters)
    actor_net = mlp(; 
        dims = [
            nn_params.shared_out_size[1], 
            fill(nn_params.neurons, nn_params.hidden_layers)..., 
            nn_params.action_size
        ],
        act_fun = nn_params.activation,
        head_init = Flux.orthogonal(; gain=sqrt(0.01)),
        head = true,
        p_dropout = nn_params.p_dropout
    )
    DiscreteActor(actor_net, nn_params.actor_loss)
end

function (actor::DiscreteActor)(x; logits=false) 
    @assert all(isfinite, x)
    out = logits ? actor.net(x) : softmax(actor.net(x); dims=1)
    @assert all(isfinite, out)
    return out
end

function getloss(actor::DiscreteActor, input; policy_target)
    actor.loss(actor.net(input), policy_target)
end

"""
    Critic
"""
struct Critic{NET, L, C, OT, LT}
    net::NET
    loss::L
    categories::C # vector of labels for distributional critic (i.e. muzero/dreamer)
    output_transform::OT # transform output post network(including categories). Typically Flux.Scale
    loss_transform::LT # transforms the value target. should be inverse of output_transform
end
Flux.@layer :expand Critic trainable=(net)

function Critic(nn_params::NetworkParameters)
    # add transform paramter later
    # add warning for categories but mse/mae loss (and functionality?)
    output_transform = identity
    loss_transform = identity # inverse of output_transform
    if nn_params.critic_loss ∈ [Flux.Losses.mse, Flux.Losses.mae]
        if !isempty(nn_params.critic_categories)
            @warn "critic_categories provided but not used"
        end
        critic_head_size = 1
        critic_head_gain = 1
    elseif nn_params.critic_loss == Flux.Losses.logitcrossentropy
        @assert !isempty(nn_params.critic_categories) "Must provide NetworkParameters.critic_categories"
        critic_head_size = length(nn_params.critic_categories)
        critic_head_gain = 0.01
    else
        @assert false "Critic loss $(nn_params.critic_loss) is not supported. Must use mse, mae, or logitcrossentropy."
    end
    critic_net = mlp(; 
        dims = [
            nn_params.shared_out_size[1], 
            fill(nn_params.neurons, nn_params.hidden_layers)..., 
            critic_head_size
        ],
        act_fun = nn_params.activation,
        head_init = Flux.orthogonal(; gain=sqrt(critic_head_gain)),
        head = true,
        p_dropout = nn_params.p_dropout
    )
    Critic(
        critic_net, 
        nn_params.critic_loss, 
        Float32.(nn_params.critic_categories),
        output_transform, 
        loss_transform
    )
end

function (critic::Critic)(x)
    @assert all(isfinite, x)
    net_out = x |> critic.net
    nominal_value = size(net_out, 1) == 1 ? net_out : critic.categories' * softmax(net_out; dims=1)
    out = nominal_value |> critic.output_transform
    @assert all(isfinite, net_out) "$x"
    @assert all(isfinite, nominal_value)
    @assert all(isfinite, out)
    return out
end

function getloss(critic::Critic, input; value_target)
    net_out = critic.net(input)
    target_scalar = Flux.Zygote.@ignore_derivatives critic.loss_transform(value_target)
    if size(net_out,1) == 1
        target = target_scalar
    else
        target = Flux.Zygote.@ignore_derivatives twohot(value_target, critic.categories)
    end
    loss = critic.loss(net_out, target)

    # make sure to get MSE to calculate fraction variance unexplained
    mse = Flux.Zygote.ignore_derivatives() do 
        critic.loss == Flux.Losses.mse && return loss

        nominal_value = size(net_out, 1) == 1 ? net_out : critic.categories' * softmax(net_out; dims=1)
        val_est = nominal_value |> critic.output_transform    
        Flux.Losses.mse(val_est, target_scalar)
    end

    return loss, mse
end

"""
    twohot
"""
twohot(x, labels::CuVector; kwargs...) = cu(twohot(Array(x), Array(labels); kwargs...))
function twohot(x, labels::Vector; kwargs...)
    @assert ndims(x) == 2 "error: ndims=$(ndims(x)), type = $(typeof(x))"
    @assert size(x,1) == 1 || size(x,2) == 1 "x must be a row or column vector"
    twohot(vec(x), labels)
end
function twohot(x::Union{Real,AbstractVector}, labels::Vector; clamping=true)
    # returns sparse n_labels x n_x
    @assert clamping "Clamping turned off is a future feature lol"
    # @assert issorted(labels) "Labels must be sorted"
    
    x = clamping ? clamp.(x, labels[1], labels[end]) : x
    ub_i = map(y->searchsortedfirst(labels, y), x)
    clamp!(ub_i, 2, length(labels)) # each entry will have a distinct ub and a lb
    lb_i = ub_i .- 1
    ub = labels[ub_i]
    lb = labels[lb_i]
    a = (x .- lb) ./ (ub .- lb) # x = (1-a)*lb + a*ub
    rows = [lb_i'; ub_i']
    vals = [1 .- a'; a']
    cols = [(1:length(x))'; (1:length(x))']
    sparse(vec(rows), vec(cols), vec(vals), length(labels), length(x))    
end

"""
    mlp 
"""
function mlp(;
    dims, 
    act_fun = tanh, 
    hidden_init = Flux.orthogonal(; gain=sqrt(2)),
    head_init = hidden_init,
    head = true,
    p_dropout = 0
    )
    end_idx = head ? length(dims)-1 : length(dims)
    layers = Any[]
    for ii = 1:end_idx-1
        push!(layers, Dense(dims[ii] => dims[ii+1], act_fun; init=hidden_init))
        if !iszero(p_dropout)
            push!(layers, Dropout(p_dropout))
        end
    end
    head && push!(layers, Dense(dims[end-1] => dims[end]; init=head_init))
    return Chain(layers...)
end

end