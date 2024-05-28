

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

function MCTS.estimate_value(heuristic::NetworkWrapper, pomdp::POMDP{S}, b::PFTBelief{S}, depth::Int) where {S}
    get_value(heuristic, b)
end

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
    enable_action_pw::Bool
end
function PUCT(; net, c=1.0)
    function(sol::PFTDPWSolver, pomdp::POMDP)
        A = actions(pomdp)
        Ai = map(a->actionindex(pomdp, a), A)
        ordered_actions = A[sortperm(Ai)]
        PUCT(net, pomdp, ordered_actions, sol.k_a, sol.alpha_a, Float64(c), sol.enable_action_pw)
    end
end
function ParticleFilterTrees.select_action(criteria::PUCT, tree::PFTDPWTree{S,A}, b_idx) where {S,A}
    (; net, pomdp, ordered_actions, k, alpha, c, enable_action_pw) = criteria

    P = get_policy(net, tree.b[b_idx])

    if enable_action_pw && length(tree.b_children[b_idx]) ≤ k*tree.Nh[b_idx]^alpha
        a = ordered_actions[sample_cat(P)]
        ParticleFilterTrees.insert_action!(tree, b_idx, a)
    elseif !enable_action_pw && isempty(tree.b_children[b_idx])
        for a in actions(pomdp)
            ParticleFilterTrees.insert_action!(tree, b_idx, a)
        end
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

function completed_policy(tree::PFTDPWTree{S,A}, b_idx, V, logits, pomdp) where {S,A}
    # completed_Q = N(a) > 0 ? Q(a) : V(a)
    completed_Q = copy(V)
    for (a, a_idx) in tree.b_children[b_idx]
        completed_Q[actionindex(pomdp, a)] = tree.Qha[a_idx]
    end

    sigma_Q = sigma(Q)

    new_policy = softmax(logits + sigma_Q)
end

function nonroot_action_selction()
    new_policy = completed_policy()
    N = tree.Nh[b_idx]

    objective = copy(new_policy)

    for (a, _) in tree.b_children[b_idx]
        Na = tree.Nha[a_idx]
        pomdp_a_idx = actionindex(pomdp, a)
        objective[pomdp_a_idx] -= Na / N
    end

    opt_pomdp_a_idx = argmax(objetive)
    
    opt_a = ordered_actions[opt_pomdp_a_idx]
    opt_ba_idx = get_ba_idx!(b_idx, opt_a)

    return opt_a => opt_ba_idx
end



