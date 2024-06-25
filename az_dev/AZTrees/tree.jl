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
- `prior_logits` - Logits of state node (prior)
- `improved_policy` - Posteriro policy of state node. Used as a temporary variable
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
    prior_logits::Matrix{Float32}
    improved_policy::Vector{Float32} # recalculated each node, never saved

    function GuidedTree{S,A}(sz::Int, na::Int, k_o::Int) where {S, A}
        return new{S, A}(
            PushVector{Int}(sz),
            PushVector{Int}(sz),
            PushVector{Float64}(sz),

            PushVector{S}(sz),
            PushVector{Float64}(sz),

            NestedPushVector{Pair{A,Int}}(k_o, sz),
            NestedPushVector{Int}(na, sz),
            PushVector{Int}(sz),
            PushVector{Int}(sz),

            PushVector{Float64}(sz),
            Matrix{Float32}(undef, na, sz),
            Vector{Float32}(undef, na),
        )
    end
end

function reset_tree!(tree::GuidedTree)
    empty!(tree.Nh)
    empty!(tree.Nha)
    empty!(tree.Qha)
    empty!(tree.s)
    empty!(tree.s_rewards)
    empty!(tree.s_children)
    empty!(tree.sa_children)
    empty!(tree.s_parent)
    empty!(tree.sa_parent)
    empty!(tree.prior_value)
    return nothing
end

function insert_state!(tree::GuidedTree{S}, s::S, sa_idx::Int, r::Real) where S
    s_idx = 1 + length(tree.s)

    push!(tree.s, s)
    push!(tree.Nh, 0)
    push!(tree.s_rewards, r)
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
