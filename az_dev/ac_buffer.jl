struct ACBatch{T, N, Q_GPU <: CuArray}
    batchsize       :: Int
    cpu_querry      :: Array{T,N}
    gpu_querry      :: Q_GPU
    value           :: Vector{T}
    policy          :: Matrix{T}
    agent_idxs      :: Vector{Int}
    querry_ready    :: Threads.Event
    response_ready  :: Threads.Event
    ready_count     :: Threads.Atomic{Int64}

    function ACBatch{T}(batchsize::Int, in_size::NTuple{N,Int}, na::Int) where {T, N}
        q_gpu = CuArray{T}(undef, in_size..., batchsize)
        return new{T, N+1, typeof(q_gpu)}(
            batchsize,
            Array{T}(undef, in_size..., batchsize) |> CUDA.pin,
            q_gpu,
            Vector{T}(undef, batchsize) |> CUDA.pin,
            Matrix{T}(undef, na, batchsize) |> CUDA.pin,
            Vector{Int}(undef, batchsize),
            Threads.Event(),
            Threads.Event(),
            Threads.Atomic{Int}(0)
        )
    end
end

ACBatch(batchsize, in_size, na) = ACBatch{Float32}(batchsize, in_size, na)

struct BatchManager{B <: ACBatch}
    batchsize       :: Int
    batches         :: Vector{B}
    querry_idx      :: Threads.Atomic{Int64}
    response_idx    :: Threads.Atomic{Int64}
    gpu_batch_idx   :: Threads.Atomic{Int64}
end

# Must make sure there are more batches than can be filled with querries at once
# n_batches = ceil(Int, 1 + n_agents / batchsize)
function BatchManager(n_batches::Int, batchsize::Int, in_size::NTuple, na::Int)
    return BatchManager(
        batchsize,
        [ACBatch(batchsize, in_size, na) for _ in 1:n_batches],
        Threads.Atomic{Int}(1),
        Threads.Atomic{Int}(1),
        Threads.Atomic{Int}(1)
    )
end

function set_querry!(bm::BatchManager, agent_idx::Int, querry)
    (; batchsize, batches, querry_idx) = bm

    batch_idx, local_idx = get_querryindices!(querry_idx, batchsize, length(batches))

    batch = batches[batch_idx]

    batch.cpu_querry[:, local_idx] .= querry
    batch.agent_idxs[local_idx] = agent_idx

    n_ready = 1 + Threads.atomic_add!(batch.ready_count, 1)
    if n_ready == batch.batchsize
        Threads.notify(batch.querry_ready)
    end

    return nothing
end

function get_response!(bm::BatchManager)
    (; batchsize, batches, response_idx) = bm

    batch_idx, local_idx = get_querryindices!(response_idx, batchsize, length(batches))

    batch = batches[batch_idx]

    wait(batch.response_ready)

    agent_idx = batch.agent_idxs[local_idx]
    value     = batch.value[local_idx]
    policy    = @view batch.policy[:, local_idx]

    return agent_idx, value, policy
end

function get_querryindices!(atomic_index::Threads.Atomic{Int64}, batchsize::Int, n_batches::Int)
    global_idx = Threads.atomic_add!(atomic_index, 1)
    batch_idx  = mod1(1 + (global_idx - 1) ÷ batchsize, n_batches)
    local_idx  = mod1(global_idx, batchsize)
    return batch_idx, local_idx
end

function process_batch!(bm::BatchManager, ac)
    (; batches, gpu_batch_idx) = bm

    batch_idx  = mod1(gpu_batch_idx[], length(batches))
    batch      = batches[batch_idx]

    wait(batch.querry_ready)

    copyto!(batch.gpu_querry, batch.cpu_querry)

    (; policy, value) = ac(batch.gpu_querry)

    copyto!(batch.policy, policy)
    copyto!(batch.value,  value)

    CUDA.synchronize() # is this necessary?

    Threads.atomic_xchg!(batch.ready_count, 0)
    notify(batch.response_ready)
    reset(batch.querry_ready)
    Threads.atomic_add!(gpu_batch_idx, 1)

    return nothing
end

querry_ready(batch::ACBatch) = @atomic batch.querry_ready.set
response_ready(batch::ACBatch) = @atomic batch.response_ready.set

function querry_ready(bm::BatchManager)
    (; batches, gpu_batch_idx) = bm
    batch_idx  = mod1(gpu_batch_idx[], length(batches))
    batch      = batches[batch_idx]
    return querry_ready(batch)
end

function response_ready(bm::BatchManager)
    n_processed = bm.batchsize * (bm.gpu_batch_idx[] - 1)
    return bm.response_idx[] <= n_processed
end
