"""
...
- `Nh` - Number of times history node has been visited
- `Nha` - Number of times action node has been visited
- `Qha` - Q value associated with some action node
- `b` - Vector of beliefs (`PFTBelief`)
- `b_children` - Mapping belief ID to (action, action ID) pair
- `b_rewards` - R(b,a) where index is ID of b' where b' = τ(b,a,o)
- `bao_children` - `(ba_idx,O) => bp_idx`
- `ba_children` - `ba_idx => [bp_idx, bp_idx, bp_idx, ...]`
...
"""
struct GuidedTree{S,A,O}
    Nh::PV{Int}
    Nha::PV{Int}# Number of times a history-action node has been visited
    Qha::PV{Float64} # Map ba node to associated Q value

    b::PV{PFTBelief{S}}
    b_children::NPV{Pair{A,Int}}# b_idx => [(a,ba_idx), ...]
    b_rewards::PV{Float64}# Map b' node index to immediate reward associated with trajectory bao where b' = τ(bao)
    b_ntprob::PV{Float64} # Map b' node index to proportion of non-terminal states before resampling
    b_V::PV{Float32} # Map b' node index to estimated value (i.e. from network)
    b_P::PV{Vector{Float32}} # Map b' node index to estimated value (i.e. from network)

    bao_children::Dict{Tuple{Int,O},Int} # (ba_idx,O) => bp_idx
    ba_children::NPV{Int} # ba_idx => [bp_idx, bp_idx, bp_idx, ...]

    function GuidedTree{S,A,O}(sz::Int, na::Int, check_repeat_obs::Bool=true, k_o=10) where {S,A,O}
        return new(
            PushVector{Int}(sz),
            PushVector{Int}(sz),
            PushVector{Float64}(sz),

            PushVector{PFTBelief{S}}(sz),
            NestedPushVector{Pair{A,Int}}(ceil(Int,k_o), sz),
            PushVector{Float64}(sz),
            PushVector{Float64}(sz),
            PushVector{Float32}(sz),
            PushVector{Vector{Float32}}(sz),

            sizehint!(Dict{Tuple{Int,O},Int}(), check_repeat_obs ? sz : 0),
            NestedPushVector{Int}(na, sz),
        )
    end
end

function insert_belief!(
    tree::GuidedTree{S,A,O}, 
    bp::PFTBelief{S}, 
    ba_idx::Int, 
    obs::O, 
    r::Float64, 
    nt_prob::Float64, 
    V::Float32, 
    P::Vector{Float32},
    check_repeat_obs::Bool
    ) where {S,A,O}
    
    bp_idx = _insert_belief!(tree, bp, V, P, r, nt_prob)
    push!(tree.ba_children[ba_idx], bp_idx)

    if check_repeat_obs
        tree.bao_children[(ba_idx,obs)] = bp_idx
    end

    return bp_idx
end

function _insert_belief!(
    tree::GuidedTree{S,A,O}, 
    b::PFTBelief{S}, 
    V::Float32,
    P::Vector{Float32},
    r::Float64 = 0.0,
    nt_prob::Float64 = 1.0
    ) where {S,A,O}

    b_idx = length(tree.b)+1
    push!(tree.b, b)
    push!(tree.Nh, 0)
    push!(tree.b_rewards, r)
    push!(tree.b_ntprob, nt_prob)
    push!(tree.b_V, V)
    push!(tree.b_P, P)
    freenext!(tree.b_children)
    return b_idx
end

function insert_root!(
    tree::GuidedTree{S,A}, 
    b::PFTBelief{S},
    V::Float32,
    P::Vector{Float32}
    ) where {S,A}

    empty!(tree.Nh)
    empty!(tree.Nha)
    empty!(tree.Qha)
    empty!(tree.b)
    empty!(tree.b_children)
    empty!(tree.b_rewards)
    empty!(tree.bao_children)
    empty!(tree.ba_children)
    empty!(tree.b_ntprob)
    empty!(tree.b_V)
    empty!(tree.b_P)

    _insert_belief!(tree, b, V, P)
end

function insert_action!(
    tree::GuidedTree{S,A}, 
    b_idx::Int, 
    a::A, 
    Q_init::Float64 = 0.0, 
    Na_init::Int = 0
    ) where {S,A}

    for (_a, ba_idx) in tree.b_children[b_idx]
        _a == a && return ba_idx
    end

    return _insert_action!(tree, b_idx, a, Q_init, Na_init)
end

function _insert_action!(
    tree::GuidedTree{S,A}, 
    b_idx::Int, 
    a::A, 
    Q_init::Float64, 
    Na_init::Int
    ) where {S,A}

    ba_idx = length(tree.ba_children)+1
    push!(tree.b_children[b_idx], a=>ba_idx)
    freenext!(tree.ba_children)
    push!(tree.Nha, Na_init)
    push!(tree.Qha, Q_init)

    return ba_idx
end