"""
...
- `Nh` - Number of times history node has been visited
- `Nha` - Number of times action node has been visited
- `Qha` - Q value associated with some action node
- `s` - MDP State associated with some history node
- `rewards` - R(s,a,s') where index is ID of s' where s' = τ(s,a,o)
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
    Nh              :: Vector{Int}
    Nha             :: Vector{Int}
    Qha             :: Vector{Float64}

    s               :: Vector{S}
    rewards         :: Vector{Float64}

    s_children      :: Vector{Vector{Tuple{A, Int}}}
    sa_children     :: Vector{Vector{Int}}

    s_parent        :: Vector{Int}
    sa_parent       :: Vector{Int}

    prior_value     :: Vector{Float64}
    prior_logits    :: Matrix{Float32}
    improved_policy :: Vector{Float32} # recalculated each node, never saved

    function GuidedTree{S,A}(sz::Int, na::Int, k_o::Int) where {S, A}
        Nh = Vector{Int}()
        Nha = Vector{Int}()
        Qha = Vector{Float64}()

        s = Vector{S}()
        rewards = Vector{Float64}()

        s_children = [Vector{Tuple{A, Int}}() for _ in 1:sz]
        sa_children = [Vector{Int}() for _ in 1:sz]

        s_parent = Vector{Int}()
        sa_parent = Vector{Int}()

        prior_value = Vector{Float64}()
        prior_logits = Matrix{Float32}(undef, na, sz)
        improved_policy = Vector{Float32}(undef, na)

        for x in (Nh, Nha, Qha, s, rewards, s_children, s_parent, sa_parent, prior_value)
            sizehint!(x, sz)
        end

        for x in s_children
            sizehint!(x, k_o)
        end

        for x in sa_children
            sizehint!(x, na)
        end

        return new{S, A}(
            Nh,
            Nha,
            Qha,

            s,
            rewards,

            s_children,
            sa_children,

            s_parent,
            sa_parent,

            prior_value,
            prior_logits,
            improved_policy
        )
    end
end

function reset_tree!(tree::GuidedTree)
    (; Nh, Nha, Qha, s, rewards, s_children, sa_children, s_parent, sa_parent, prior_value) = tree
    foreach(empty!, (Nh, Nha, Qha, s, rewards, s_parent, sa_parent, prior_value))
    foreach(empty!, s_children)
    foreach(empty!, sa_children)
    # GC.gc(false)
    return
end

function insert_state!(tree::GuidedTree{S}, s::S, sa_idx::Int, r::Real) where S
    s_idx = 1 + length(tree.s)

    push!(tree.s, s)
    push!(tree.Nh, 0)
    push!(tree.rewards, r)
    push!(tree.sa_parent, sa_idx)

    iszero(sa_idx) || push!(tree.sa_children[sa_idx], s_idx)

    return s_idx
end

function insert_action!(tree::GuidedTree{S,A}, s_idx::Int, a::A;
    Q_init::Real = 0.0,
    Na_init::Int = 0
    ) where {S,A}

    for (_a, sa_idx) in tree.s_children[s_idx]
        _a == a && return sa_idx
    end

    sa_idx = 1 + length(tree.Nha)

    push!(tree.s_children[s_idx], (a, sa_idx))
    push!(tree.Nha, Na_init)
    push!(tree.Qha, Q_init)
    push!(tree.s_parent, s_idx)

    return sa_idx
end
