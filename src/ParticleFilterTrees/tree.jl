"""
...
- `Nh` - Number of times history node has been visited
- `Nha` - Number of times action node has been visited
- `Qha` - Q value associated with some action node
- `b` - Vector of beliefs (`PFTBelief`)
- `b_children` - Mapping belief ID to (action, action ID) pair
- `b_rewards` - R(b,a) where index is ID of b' where b' = τ(b,a,o)
- `ba_children` - `ba_idx => [bp_idx, bp_idx, bp_idx, ...]`
...
"""
struct GuidedTree{S, A}
    Nh::PV{Int}
    Nha::PV{Int}# Number of times a history-action node has been visited
    Qha::PV{Float64} # Map ba node to associated Q value

    b::PV{S}
    b_children::NPV{Pair{A, Int}} # b_idx => [(a,ba_idx), ...]
    b_rewards::PV{Float64} # Map b' node index to immediate reward associated with trajectory bao where b' = τ(bao)
    b_V::PV{Float64} # Map b' node index to estimated value (i.e. from network)
    b_P::PV{Vector{Float32}} # Map b' node index to policy
    b_logits::PV{Vector{Float32}} # Map b' node index to policy logits
    ba_children::NPV{Int} # ba_idx => [bp_idx, bp_idx, bp_idx, ...]

    ba_parent::PV{Int}
    b_parent::PV{Int}

    function GuidedTree{S,A}(sz::Int, na::Int, k_o=10) where {S, A}
        return new(
            PushVector{Int}(sz),
            PushVector{Int}(sz),
            PushVector{Float64}(sz),

            PushVector{S}(sz),
            NestedPushVector{Pair{A,Int}}(ceil(Int, k_o), sz),
            PushVector{Float64}(sz),
            PushVector{Float64}(sz),
            PushVector{Vector{Float32}}(sz),
            PushVector{Vector{Float32}}(sz),
            NestedPushVector{Int}(na, sz),

            PushVector{Int}(sz),
            PushVector{Int}(sz)
        )
    end
end

function reset_tree!(tree::GuidedTree)
    foreach(propertynames(tree)) do name
        empty!(getfield(tree, name))
    end
end

function insert_belief!(tree::GuidedTree{S}, b::S;
    logits::Vector{<:Real} = log.(policy),
    policy::Vector{<:Real} = softmax(logits),
    value::Real = 0.0,
    r::Real     = 0.0,
    ba_idx::Int = 0 # parent, 0 for root
    ) where S

    b_idx = 1 + length(tree.b)

    push!(tree.b, b)
    push!(tree.Nh, 0)
    push!(tree.b_rewards, r)
    push!(tree.b_V, value)
    push!(tree.b_logits, logits)
    push!(tree.b_P, policy)
    push!(tree.ba_parent, ba_idx)
    iszero(ba_idx) || push!(tree.ba_children[ba_idx], b_idx)

    freenext!(tree.b_children)

    return b_idx
end

function insert_action!(tree::GuidedTree{S,A}, b_idx::Int, a::A;
    Q_init::Real = 0.0,
    Na_init::Int = 0
    ) where {S,A}

    for (_a, ba_idx) in tree.b_children[b_idx]
        _a == a && return ba_idx
    end

    ba_idx = 1 + length(tree.ba_children)

    push!(tree.b_children[b_idx], a=>ba_idx)
    push!(tree.Nha, Na_init)
    push!(tree.Qha, Q_init)
    push!(tree.b_parent, b_idx)

    freenext!(tree.ba_children)

    return ba_idx
end
