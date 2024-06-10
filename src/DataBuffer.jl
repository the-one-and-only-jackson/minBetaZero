@kwdef mutable struct DataBuffer{A,B,C}
    const network_input::A
    const value_target::B
    const policy_target::C
    const capacity::Int
    length::Int = 0
    idx::Int = 1
    const minib_idxs::Vector{Int} = Int[]
    minib_pointer::Int = 0
end

function reset_buffer!(b::DataBuffer)
    b.length = 0
    b.idx = 1
    empty!(b.minib_idxs)
    b.minib_pointer = 0
end

function reset_minibatch(b::DataBuffer)
    shuffle!(b.minib_idxs)
    b.minib_pointer = 0
end

function sample_minibatch(b::DataBuffer, batchsize::Int)
    @assert batchsize <= b.length "batchsize: $batchsize, length: $(b.length)"

    L1 = min(batchsize, b.length - b.minib_pointer) 
    L2 = batchsize - L1
    minib_idxs_1 = b.minib_idxs[(1:L1) .+ b.minib_pointer]
    if L2 != 0
        shuffle!(b.minib_idxs)
        minib_idxs_2 = b.minib_idxs[1:L2]
        minib_idxs = vcat(minib_idxs_1, minib_idxs_2)
    else
        if b.minib_pointer + batchsize == b.length
            shuffle!(b.minib_idxs)
        end
        minib_idxs = minib_idxs_1
    end
    b.minib_pointer = mod(b.minib_pointer + batchsize, b.length)

    data = (;
        network_input = select_last_dim(b.network_input, minib_idxs),
        value_target  = select_last_dim(b.value_target , minib_idxs),
        policy_target = select_last_dim(b.policy_target, minib_idxs)
    )

    return map(copy, data) # make everything stride 1 for network
end

function DataBuffer(input_dims::Tuple{Vararg{Int}}, na::Int, capacity::Int)
    network_input = zeros(Float32, input_dims..., capacity)
    value_target  = zeros(Float32, 1, capacity)
    policy_target = zeros(Float32, na, capacity)
    return DataBuffer(; network_input, value_target, policy_target, capacity)
end

function set_buffer!(b::DataBuffer; network_input, value_target, policy_target)
    @assert (size(network_input)[1:end-1] == size(b.network_input)[1:end-1]) || (size(network_input) == size(b.network_input)[1:end-1]) "$(size(network_input)), $(size(b.network_input))"
    @assert (size(value_target)[1:end-1] == size(b.value_target)[1:end-1]) || (size(value_target ) == size(b.value_target )[1:end-1]) "value $(size(value_target))"
    @assert (size(policy_target)[1:end-1] == size(b.policy_target)[1:end-1]) || (size(policy_target) == size(b.policy_target)[1:end-1])

    L = length(value_target)

    if b.capacity - b.idx + 1 >= L
        dst_idxs = b.idx .+ (0:L-1)
        copyto!(select_last_dim(b.network_input, dst_idxs), network_input)
        copyto!(select_last_dim(b.value_target, dst_idxs), value_target)
        copyto!(select_last_dim(b.policy_target, dst_idxs), policy_target)
    else
        L1 = b.capacity - b.idx + 1
        dst_idxs = b.idx:b.capacity
        src_idxs = 1:L1
        copyto!(select_last_dim(b.network_input, dst_idxs), select_last_dim(network_input, src_idxs))
        copyto!(select_last_dim(b.value_target, dst_idxs), select_last_dim(value_target, src_idxs))
        copyto!(select_last_dim(b.policy_target, dst_idxs), select_last_dim(policy_target, src_idxs))

        L2 = L - L1
        dst_idxs = 1:L2
        src_idxs = L1+1:L
        copyto!(select_last_dim(b.network_input, dst_idxs), select_last_dim(network_input, src_idxs))
        copyto!(select_last_dim(b.value_target, dst_idxs), select_last_dim(value_target, src_idxs))
        copyto!(select_last_dim(b.policy_target, dst_idxs), select_last_dim(policy_target, src_idxs))
    end
    
    b.idx = mod1(b.idx + L, b.capacity)

    if b.length < b.capacity
        b.length = min(b.length + L, b.capacity)

        new_idxs = length(b.minib_idxs)+1:b.length
        append!(b.minib_idxs, new_idxs)
    end

    return nothing
end

@inline select_last_dim(arr, idxs) = selectdim(arr, ndims(arr), idxs)
