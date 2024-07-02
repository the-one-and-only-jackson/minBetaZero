"""
...
- `Nh` - Number of times history node has been visited
- `Nha` - Number of times action node has been visited
- `Qha` - Q value associated with some action node
- `states` - MDP State associated with some history node
- `reward` - R(s,a,s') where index is ID of s'
- `s_children` - Mapping state ID to (action, action ID) pair
- `sa_children` - `sa_idx => [bp_idx, bp_idx, bp_idx, ...]`
- `prior_value` - Value of state node (prior)
- `prior_logits` - Logits of state node (prior)
...
"""
mutable struct GuidedTree{S, TInt, TFloat}
    const Nh            :: Vector{TInt}
    const Nha           :: Vector{TInt}
    const Qha           :: Vector{TFloat}
    const reward        :: Vector{TFloat}
    const s_children    :: Matrix{TInt}
    const sa_children   :: Matrix{TInt}
    const prior_value   :: Vector{TFloat}
    const prior_logits  :: Matrix{TFloat}
    const node_stack    :: Vector{TInt} # [sa, s, ...]

    state       :: Vector{S}

    stack_index :: TInt
    s_counter   :: TInt
    sa_counter  :: TInt
    qmin        :: TFloat
    qmax        :: TFloat

    function GuidedTree{S, TInt, TFloat}(sz, na, k_o) where {S, TInt, TFloat}
        new{S, TInt, TFloat}(
            zeros(TInt, sz),
            zeros(TInt, sz),
            zeros(TFloat, sz),
            zeros(TFloat, sz),
            zeros(TInt, na, sz),
            zeros(TInt, k_o, sz),
            zeros(TFloat, sz),
            zeros(TFloat, na, sz),
            zeros(TInt, 2*sz),

            Vector{S}(undef, sz),

            zero(TInt),
            zero(TInt),
            zero(TInt),
            typemax(TFloat),
            typemin(TFloat)
        )
    end

    function GuidedTree{S}(sz, na, k_o) where {S}
        return GuidedTree{S, Int64, Float64}(sz, na, k_o)
    end
end

function reset!(tree::GuidedTree)
    fill!(tree.Nh , 0)
    fill!(tree.Nha, 0)
    fill!(tree.Qha, 0)
    fill!(tree.s_children, 0)
    fill!(tree.sa_children, 0)

    tree.state = Vector{eltype(tree.state)}(undef, length(tree.state))

    tree.s_counter  = 0
    tree.sa_counter = 0
    tree.qmin = typemax(tree.qmin)
    tree.qmax = typemin(tree.qmin)

    return nothing
end

function insert_state!(tree::GuidedTree{S}, s::S, sa_idx::Integer = 0, r::Real = 0) where S
    s_idx = tree.s_counter += one(tree.s_counter)

    tree.state[s_idx]  = s
    tree.reward[s_idx] = r

    if !iszero(sa_idx) # if not root
        index = findfirst(iszero, @view tree.sa_children[:, sa_idx])
        @assert !isnothing(index) "sa_children at sa_idx = $sa_idx is full"
        tree.sa_children[index, sa_idx] = s_idx
    end

    return s_idx
end

function insert_action!(tree::GuidedTree, s_idx::Integer, ai::Integer)
    sa_idx = tree.s_children[ai, s_idx]
    if iszero(sa_idx)
        sa_idx = tree.sa_counter += one(tree.sa_counter)
        tree.s_children[ai, s_idx] = sa_idx
    end
    return sa_idx
end

function update_prior!(tree::GuidedTree, s_idx::Integer, logits, value)
    tree.prior_logits[:, s_idx] .= logits
    tree.prior_value[s_idx] = value
    update_dq!(tree, value)
    return nothing
end

function update_dq!(tree::GuidedTree, q)
    if q < tree.qmin
        tree.qmin = q
    end
    if q > tree.qmax
        tree.qmax = q
    end
    return nothing
end

function get_dq(tree::GuidedTree, s_idx::Integer; eps=1e-6, global_dq::Bool=true)
    if global_dq
        qmin = tree.qmin
        qmax = tree.qmax
    else
        qmin = tree.prior_value[s_idx]
        qmax = tree.prior_value[s_idx]

        for (_, sa_idx) in s_children(tree, s_idx)
            Q = tree.Qha[sa_idx]
            if Q < qmin
                qmin = Q
            elseif Q > qmax
                qmax = Q
            end
        end
    end

    dq = qmax - qmin
    dq = dq < eps ? one(dq) : dq

    return dq
end

stack_empty(tree::GuidedTree) = iszero(tree.stack_index)

function push_stack!(tree::GuidedTree, sa_idx::Integer, s_idx::Integer)
    tree.stack_index += 1
    tree.node_stack[tree.stack_index] = sa_idx
    tree.stack_index += 1
    tree.node_stack[tree.stack_index] = s_idx
    return nothing
end

function pop_stack!(tree::GuidedTree)
    if tree.stack_index < 2
        dump(tree)
    end
    s_idx = tree.node_stack[tree.stack_index]
    tree.stack_index -= 1
    sa_idx = tree.node_stack[tree.stack_index]
    tree.stack_index -= 1
    return sa_idx, s_idx
end

function s_children(tree::GuidedTree, s_idx::Integer)
    Iterators.filter(
        (ai, sa_idx)::Tuple -> !iszero(sa_idx),
        enumerate(@view tree.s_children[:, s_idx])
    )
end

function n_s_children(tree::GuidedTree, s_idx::Integer)
    count(!iszero, @view tree.s_children[:, s_idx])
end

function sa_children(tree::GuidedTree, sa_idx::Integer)
    Iterators.filter(!iszero, @view tree.sa_children[:, sa_idx])
end

function n_sa_children(tree::GuidedTree, sa_idx::Integer)
    count(!iszero, @view tree.sa_children[:, sa_idx])
end
