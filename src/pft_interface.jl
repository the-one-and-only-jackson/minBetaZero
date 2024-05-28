

function input_representation end

"""
Get belief representation for network input, add batch dimension for Flux.

Note: Only looking at batch = 1 (MCTS)!
"""
function network_input(belief)
    b = convert(AbstractArray{Float32}, input_representation(belief))
    return Flux.unsqueeze(b; dims=ndims(b)+1) # add extra single dimension (batch = 1)
end

@kwdef struct NetworkWrapper{N<:ActorCritic}
    net::N
    input::Vector{Any} = Any[] # gross, fix later
    value::Vector{Float32} = Float32[]
    policy::Vector{Vector{Float32}} = Vector{Float32}[]
end
NetworkWrapper(net) = NetworkWrapper(; net)

function insert_belief!(heuristic::NetworkWrapper, b; logits=false)
    idx = findfirst(==(b), heuristic.input)
    if isnothing(idx)
        pv = heuristic.net(network_input(b); logits)
        p = vec(pv.policy)
        v = pv.value[]
        push!(heuristic.input, b)
        push!(heuristic.value, v)
        push!(heuristic.policy, p)
        idx = length(heuristic.input)
    end
    idx
end

get_value(heuristic::NetworkWrapper, b) = heuristic.value[insert_belief!(heuristic, b)]
get_policy(heuristic::NetworkWrapper, b; logits=false) = heuristic.policy[insert_belief!(heuristic, b; logits)]

MCTS.estimate_value(heuristic::NetworkWrapper, pomdp, b, depth) = get_value(heuristic, b)

function Base.empty!(heuristic::NetworkWrapper)
    empty!(heuristic.input)
    empty!(heuristic.value)
    empty!(heuristic.policy)
    heuristic
end


struct PUCT{N<:NetworkWrapper,P,A}
    net::N
    pomdp::P
    ordered_actions::A
    k::Float64
    alpha::Float64
    c::Float64
end
function PUCT(; net, c=1.0)
    function(sol::PFTDPWSolver, pomdp::POMDP)
        A = actions(pomdp)
        Ai = map(a->actionindex(pomdp, a), A)
        ordered_actions = A[sortperm(Ai)]
        PUCT(net, pomdp, ordered_actions, sol.k_a, sol.alpha_a, Float64(c))
    end
end
function ParticleFilterTrees.select_action(criteria::PUCT, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    (; net, pomdp, ordered_actions, k, alpha, c) = criteria

    P = get_policy(net, tree.b[b_idx])

    if length(tree.b_children[b_idx]) ≤ k*tree.Nh[b_idx]^alpha
        a = ordered_actions[sample_cat(P)]
        ParticleFilterTrees.insert_action!(tree, b_idx, a)
    end

    max_crit = -Inf
    local opt_a::A
    opt_idx = 0
    for (a, ba_idx) in tree.b_children[b_idx]
        ai = actionindex(pomdp, a)
        crit = tree.Qha[ba_idx] + c * P[ai] * sqrt(tree.Nh[b_idx]) / (1 + tree.Nha[ba_idx])
        if crit > max_crit
            max_crit = crit
            opt_a = a
            opt_idx = ba_idx
        end
    end

    return opt_a => opt_idx
end

function sample_cat(p)
    r = sum(p) * rand()
    s = zero(eltype(p))
    for i in eachindex(p)
        s += p[i]
        s >= r && return i
    end
    return length(p)
end

#= 
struct Gumbel{N<:NetworkWrapper}
    net::N
    pomdp::P
    ordered_actions::A
    gumbel_sample_root::Vector
    n_act_init::Int
    n_sim::Int
    a_idx_remaining::Vector{Int}
end

function ParticleFilterTrees.select_action(criteria::Gumbel, tree::PFTDPWTree{S,A}, b_idx, d) where {S,A}
    (; net, pomdp, ordered_actions, gumbel_sample_root) = criteria

    P_logits = get_policy(net, tree.b[b_idx]; logits=true)

    if d == 0
        
    else

    end

    return opt_a => opt_idx
end
=#