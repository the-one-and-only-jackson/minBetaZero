struct ACBatch{T, Q_CPU <: Array, Q_GPU <: CuArray}
    batchsize   :: Int
    cpu_querry  :: Q_CPU
    gpu_querry  :: Q_GPU
    value       :: Vector{T}
    policy      :: Matrix{T}
    agent_idxs  :: Vector{Int}
    ready_count :: Threads.Atomic{Int}

    function ACBatch{T}(batchsize::Int, in_size::NTuple{N,Int}, na::Int) where {T, N}
        q_gpu = CuArray{T}(undef, in_size..., batchsize)
        return new{T, Array{T,N+1}, typeof(q_gpu)}(
            batchsize,
            Array{T}(undef, in_size..., batchsize) |> CUDA.pin,
            q_gpu,
            Vector{T}(undef, batchsize) |> CUDA.pin,
            Matrix{T}(undef, na, batchsize) |> CUDA.pin,
            Vector{Int}(undef, batchsize),
            Threads.Atomic{Int}(0)
        )
    end
end

function set_querry!(batch::ACBatch, batchindex::Integer, agent_idx::Integer, query)
    (; cpu_querry, agent_idxs, ready_count) = batch
    cpu_querry[:, batchindex] .= query
    agent_idxs[batchindex] = agent_idx
    n_ready = 1 + Threads.atomic_add!(ready_count, 1)
    return n_ready
end

function get_response(batch::ACBatch, index::Integer)
    agent_index = batch.agent_idxs[index]
    value       = batch.value[index]
    policy      = @view batch.policy[:, index]
    return agent_index, value, policy
end

function process_batch!(batch::ACBatch, ac)
    @assert all(isfinite, batch.gpu_querry) "$batch.gpu_querry"

    (; policy, value) = ac(batch.gpu_querry; logits=true)

    @assert all(isfinite, value)
    @assert all(isfinite, policy)

    copyto!(batch.policy, policy)
    copyto!(batch.value,  value)

    CUDA.synchronize() # is this necessary?
    Threads.atomic_xchg!(batch.ready_count, 0)

    return nothing
end

struct BatchManager{T, B <: ACBatch}
    batches        :: Vector{B}
    gpu_jobs       :: Channel{B}
    avail_querries :: Channel{Tuple{B, Int}}
    batchsize      :: Int

    function BatchManager{T}(; batchsize, in_size, na, n_batches) where T
        batches  = [ACBatch{T}(batchsize, in_size, na) for _ in 1:n_batches]
        B        = eltype(batches)
        gpu_jobs = Channel{B}(n_batches)

        avail_querries = Channel{Tuple{B, Int}}(n_batches * batchsize)
        for batch in batches, batchindex in 1:batchsize
            put!(avail_querries, (batch, batchindex))
        end

        return new{T, B}(batches, gpu_jobs, avail_querries, batchsize)
    end
end

Base.eltype(::BatchManager{T, B}) where {T, B} = B
Base.isready(bm::BatchManager) = isready(bm.gpu_jobs)
Base.take!(bm::BatchManager) = take!(bm.gpu_jobs)

function set_querry!(bm::BatchManager, agentindex::Integer, query)
    batch, batchindex = take!(bm.avail_querries)
    n_ready = set_querry!(batch, batchindex, agentindex, query)

    if n_ready == batch.batchsize
        queue_batch!(bm, batch)
    end

    return nothing
end

function queue_batch!(bm::BatchManager, batch::ACBatch)
    copyto!(batch.gpu_querry, batch.cpu_querry)
    CUDA.synchronize()
    put!(bm.gpu_jobs, batch)
    return nothing
end

function free_querry!(bm::BatchManager, batch::ACBatch, index::Integer)
    put!(bm.avail_querries, (batch, index))
    return nothing
end
