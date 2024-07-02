struct Batch{IN <: AbstractArray, OUT <: AbstractMatrix}
    network_input   :: IN  # (in..., batchsize)
    value_target    :: OUT # (na   , batchsize)
    policy_target   :: OUT # (1    , batchsize)
    length          :: Int

    function Batch(in::IN, v::OUT, p::OUT, len::Integer) where {IN, OUT}
        in_bl = size(in, ndims(in))
        v_bl  = length(v)
        p_bl  = size(p, 2)
        flag  = in_bl == len && v_bl == len && p_bl == len
        @assert flag "Batchlength: $len, ($in_bl, $v_bl, $p_bl)"
        new{IN, OUT}(in, v, p, Int(len))
    end
end

function cpuBatch(in_sz::Tuple, na::Integer, batchlength::Integer)
    network_input = zeros(Float32, in_sz..., batchlength)
    value_target  = zeros(Float32, 1       , batchlength)
    policy_target = zeros(Float32, na      , batchlength)
    Batch(network_input, value_target, policy_target, batchlength)
end

function gpuBatch(in_sz::Tuple, na::Integer, batchlength::Integer)
    network_input = CUDA.zeros(Float32, in_sz..., batchlength)
    value_target  = CUDA.zeros(Float32, 1       , batchlength)
    policy_target = CUDA.zeros(Float32, na      , batchlength)
    Batch(network_input, value_target, policy_target, batchlength)
end

Base.length(b::Batch) = b.length

mutable struct DataBuffer{BCPU <: Batch, BGPU <: Batch, RNG <: AbstractRNG}
    const batch         :: BCPU
    const minibatch_cpu :: BCPU
    const minibatch_gpu :: BGPU
    const rng           :: RNG
    length              :: Int
    index               :: Int
end

function DataBuffer(mdp::MDP, capacity::Integer, batchsize::Integer, rng::AbstractRNG)
    @assert capacity >= batchsize "Capacity not be smaller than batchsize"
    na            = length(actions(mdp))
    input_dims    = size(rand(MersenneTwister(1), initialstate(mdp)))
    batch         = cpuBatch(input_dims, na, capacity )
    minibatch_cpu = cpuBatch(input_dims, na, batchsize)
    minibatch_gpu = gpuBatch(input_dims, na, batchsize)
    DataBuffer(batch, minibatch_cpu, minibatch_gpu, rng, 0, 1)
end

function to_buffer!(
        b             :: DataBuffer,
        network_input :: AbstractVector{<:AbstractArray{<:Real}},
        value_target  :: AbstractVector{<:Real},
        policy_target :: AbstractVector{<:AbstractVector{<:Real}}
    )

    f(n, v, p) = to_buffer!(b, n, v, p)
    foreach(f, network_input, value_target, policy_target)

    return nothing
end

function to_buffer!(
        b             :: DataBuffer,
        network_input :: AbstractArray{<:Real},
        value_target  :: Real,
        policy_target :: AbstractVector{<:Real}
    )

    copyto!(select_last_dim(b.batch.network_input, b.index), network_input)
    copyto!(select_last_dim(b.batch.value_target , b.index), value_target )
    copyto!(select_last_dim(b.batch.policy_target, b.index), policy_target)

    capacity = length(b.batch)
    b.index  = mod1(b.index + 1, capacity)
    b.length = min(b.length + 1, capacity)

    return nothing
end

function sample_minibatch(b::DataBuffer)
    batchsize = length(b.minibatch_cpu)
    indicies  = rand(b.rng, 1:b.length, batchsize)

    for name in (:network_input, :value_target, :policy_target)
        arr_batch = getfield(b.batch        , name)
        arr_cpu   = getfield(b.minibatch_cpu, name)
        arr_gpu   = getfield(b.minibatch_gpu, name)

        for (dst_i, src_i) in enumerate(indicies)
            src = select_last_dim(arr_batch, src_i)
            dst = select_last_dim(arr_cpu  , dst_i)
            copyto!(dst, src)
        end

        copyto!(arr_gpu, arr_cpu)
    end

    return b.minibatch_gpu
end

@inline select_last_dim(arr, idxs) = selectdim(arr, ndims(arr), idxs)
