"""
...
- `Nh` - Number of times history node has been visited
- `Nha` - Number of times action node has been visited
- `Qha` - Q value associated with some action node
- `s` - MDP State associated with some history node
- `s_rewards` - R(s,a,s') where index is ID of s' where s' = τ(s,a,o)
- `s_children` - Mapping state ID to (action, action ID) pair
- `sa_children` - `sa_idx => [bp_idx, bp_idx, bp_idx, ...]`
- `s_parent` - History node parent of action node child
- `sa_parent` - Action node parent of history node child
- `prior_value` - Value of state node (prior)
- `prior_policy` - Policy of state node (prior)
- `prior_logits` - Logits of state node (prior)
...
"""
struct GuidedTree{S, A}
    Nh::PV{Int}
    Nha::PV{Int}
    Qha::PV{Float64}

    s::PV{S}
    s_rewards::PV{Float64}

    s_children::NPV{Pair{A, Int}}
    sa_children::NPV{Int}
    s_parent::PV{Int}
    sa_parent::PV{Int}

    prior_value::PV{Float64}
    prior_policy::PV{Vector{Float32}}
    prior_logits::PV{Vector{Float32}}

    function GuidedTree{S,A}(sz::Int, na::Int, k_o::Int) where {S, A}
        return new(
            PushVector{Int}(sz),
            PushVector{Int}(sz),
            PushVector{Float64}(sz),

            PushVector{S}(sz),
            PushVector{Float64}(sz),

            NestedPushVector{Pair{A,Int}}(ceil(Int, k_o), sz),
            NestedPushVector{Int}(na, sz),
            PushVector{Int}(sz),
            PushVector{Int}(sz)

            PushVector{Float64}(sz),
            PushVector{Vector{Float32}}(sz),
            PushVector{Vector{Float32}}(sz),
        )
    end
end

function reset_tree!(tree::GuidedTree)
    foreach(propertynames(tree)) do name
        empty!(getfield(tree, name))
    end
end

function insert_belief!(tree::GuidedTree{S}, s::S;
    logits::Vector{<:Real} = log.(policy),
    policy::Vector{<:Real} = softmax(logits),
    value::Real = 0.0,
    r::Real     = 0.0,
    sa_idx::Int = 0 # parent, 0 for root
    ) where S

    s_idx = 1 + length(tree.s)

    push!(tree.s, s)
    push!(tree.Nh, 0)
    push!(tree.s_rewards, r)
    push!(tree.prior_value, value)
    push!(tree.prior_logits, logits)
    push!(tree.prior_policy, policy)
    push!(tree.sa_parent, sa_idx)
    iszero(sa_idx) || push!(tree.sa_children[sa_idx], s_idx)

    freenext!(tree.s_children)

    return s_idx
end

function insert_action!(tree::GuidedTree{S,A}, s_idx::Int, a::A;
    Q_init::Real = 0.0,
    Na_init::Int = 0
    ) where {S,A}

    for (_a, sa_idx) in tree.s_children[s_idx]
        _a == a && return sa_idx
    end

    sa_idx = 1 + length(tree.sa_children)

    push!(tree.s_children[s_idx], a=>sa_idx)
    push!(tree.Nha, Na_init)
    push!(tree.Qha, Q_init)
    push!(tree.s_parent, s_idx)

    freenext!(tree.sa_children)

    return sa_idx
end
