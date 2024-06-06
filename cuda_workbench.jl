using CUDA
using Flux
using Statistics

net = Chain(
    x -> dropdims(mean(x; dims=2); dims=2),
    Dense(2=>256, gelu),
    Dense(256=>256, gelu),
    Dense(256=>3),
) |> gpu


querry_channel = [Channel{AbstractArray}(1) for _ in 1:500]
response_channel = [Channel{AbstractArray}(1) for _ in 1:500]

tasks = [
    Threads.@spawn begin
        z = 0f0
        for _ in 1:100
            x = randn(Float32, 2, 500)
            put!(querry_channel[i], x)
            y = take!(response_channel[i])
            z += sum(y)
        end
        return z
    end
    for i in 1:500
]


# querry_mat = CUDA.pin(zeros(Float32, 2, 500, length(querry_channel)))
# response_mat = CUDA.pin(zeros(Float32, 3, length(querry_channel)))
# device_querry = gpu(querry_mat)

# CUDA.@time while !all(istaskdone, tasks)
#     idxs = isready.(querry_channel)
#     !any(idxs) && continue

#     j = 1
#     for i in 1:length(querry_channel)
#         !idxs[i] && continue
#         querry = take!(querry_channel[i])
#         selectdim(querry_mat, ndims(querry_mat), j) .= querry
#         j += 1
#     end

#     copyto!(device_querry, querry_mat)

#     results = net(device_querry)

#     copyto!(response_mat, results)

#     j = 1
#     for i in 1:length(response_channel)
#         !idxs[i] && continue
#         put!(response_channel[i], selectdim(response_mat, ndims(response_mat), j))
#         j += 1
#     end
# end

# y = similar(x) |> gpu

# x = CUDA.pin(randn(Float32, 2,50,100));
# v = selectdim(x,3,1:40);
# Base.iscontiguous(v)
# v isa Base.FastContiguousSubArray
# length(v)


# function Base.copyto!(dest::Base.FastContiguousSubArray, src::Base.FastContiguousSubArray)
#     nd = ndims(src)
#     @assert ndims(dest) == nd
#     @assert all((size(dest, i) == size(src, i)) for i in 1:nd)
#     L = length(src)
#     unsafe_copyto!(pointer(dest), pointer(src), L)
#     dest
# end

# function copyfirst(dest, src, idxs)
#     n = ndims(src)
#     dest_view = selectdim(dest, n, 1:length(idxs))
#     src_view = selectdim(src, n, idxs)
#     copyto!(dest_view, src_view)
# end

querry_mat = CUDA.pin(zeros(Float32, 2, 500, length(querry_channel)))
response_mat = CUDA.pin(zeros(Float32, 3, length(querry_channel)))
device_querry = gpu(querry_mat)

CUDA.@time while !all(istaskdone, tasks)
    idxs = isready.(querry_channel)
    !any(idxs) && continue

    j = 1
    for i in 1:length(querry_channel)
        !idxs[i] && continue
        querry = take!(querry_channel[i])
        selectdim(querry_mat, ndims(querry_mat), j) .= querry
        j += 1
    end

    L = count(idxs)
    n = ndims(device_querry)

    
    querry_mat_view = selectdim(querry_mat, n, 1:L)
    device_querry_view = selectdim(device_querry, n, 1:L)
    copyto!(device_querry_view, querry_mat_view)

    # copyfirst(device_querry, querry_mat, 1:L)
    # copyto!(device_querry, querry_mat)

    results = net(device_querry_view)
    
    copyfirst(response_mat, results, 1:L)
    # copyto!(response_mat, results)

    j = 1
    for i in 1:length(response_channel)
        !idxs[i] && continue
        put!(response_channel[i], selectdim(response_mat, ndims(response_mat), j))
        j += 1
    end
end





CUDA.@time while !all(istaskdone, tasks)
    idxs = isready.(querry_channel)
    !any(idxs) && continue

    if count(idxs) == 1
        idx = findfirst(idxs)
        querries = take!(querry_channel[idx])
        querries_batched = reshape(querries, size(querries)..., 1)
        results = net(querries_batched |> gpu) |> cpu
        put!(response_channel[idx], results)    
    else
        querries = take!.(querry_channel[idxs])
        querries_batched = querries |> stack
        results = net(querries_batched |> gpu) |> cpu
        put!.(response_channel[idxs], eachcol(results))
    end
end



using CUDA, Flux, Statistics

@inline selectlastdim(x::Array{T,1}, idxs) where T = view(x, idxs)
@inline selectlastdim(x::CuArray{T,1}, idxs) where T = view(x, idxs)
@inline selectlastdim(x::Array{T,2}, idxs) where T = view(x, :, idxs)
@inline selectlastdim(x::CuArray{T,2}, idxs) where T = view(x, :, idxs)
@inline selectlastdim(x::Array{T,3}, idxs) where T = view(x, :, :, idxs)
@inline selectlastdim(x::CuArray{T,3}, idxs) where T = view(x, :, :, idxs)

Base.copyto!(dest::CuArray, src::Base.FastContiguousSubArray) = _copydense(dest, src)
Base.copyto!(dest::Base.FastContiguousSubArray, src::CuArray) = _copydense(dest, src)
function _copydense(dest, src)
    nd = ndims(src)
    @assert ndims(dest) == nd
    @assert all((size(dest, i) == size(src, i)) for i in 1:nd)
    L = length(src)
    unsafe_copyto!(pointer(dest), pointer(src), L)
    dest
end

function worker(querry_channel, response_channel; np=500, dx=2, calls=100*50) # 100 querries x 50 steps
    z = 0f0
    for _ in 1:calls
        x = randn(Float32, dx, np)
        put!(querry_channel, x)
        y = take!(response_channel)
        z += sum(y)
    end
    return z
end

@kwdef struct dataHolder{A}
    data::A
    idxs::Vector{Int} = zeros(Int, size(data, ndims(data)))
    lk::ReentrantLock = ReentrantLock()
end

function Base.push!(d::dataHolder, x, i::Int)
    lock(d.lk)
    try
        j = findfirst(iszero, d.idxs)
        d.idxs[j] = i
        selectlastdim(d.data, j) .= x
    finally
        unlock(d.lk)
    end
    return d
end

function sendandreset!(dest::dataHolder, src::dataHolder)
    local dest_data
    local dest_idxs
    lock(src.lk)
    lock(dest.lk)
    try
        if src.idxs[1] != 0
            if src.idxs[end] == 0
                L = findfirst(iszero, src.idxs) - 1
                src_data = selectlastdim(src.data, 1:L)
                dest_data = selectlastdim(dest.data, 1:L)
                src_idxs = @view src.idxs[1:L]
                dest_idxs = @view dest.idxs[1:L]
                @views fill!(dest.idxs[L+1:end], 0)
            else
                src_data = src.data
                dest_data = dest.data
                src_idxs = src.idxs
                dest_idxs = dest.idxs
            end
            copyto!(dest_data, src_data)
            copyto!(dest_idxs, src_idxs)

            # reset src
            fill!(src_idxs, 0)

            CUDA.synchronize()
        end
    finally
        unlock(src.lk)
        unlock(dest.lk)
    end
    return dest_data, dest_idxs
end

function hasdata(d::dataHolder)
    flag = false
    lock(d.lk)
    try
        flag = d.idxs[1] != 0
    finally
        unlock(d.lk)
    end
    return flag
end

net = Chain(
    x -> dropdims(mean(x; dims=2); dims=2),
    Dense(2=>256, gelu),
    Dense(256=>256, gelu),
    Dense(256=>3),
) |> gpu

nworkers = 500
dx = 2
dy = 3
np = 500
calls = 100*50

@sync begin
    querry_channel = [Channel{AbstractArray}(1) for _ in 1:nworkers]
    response_channel = [Channel{AbstractArray}(1) for _ in 1:nworkers]
    gpu_results_channel = Channel()

    querry_struct = dataHolder(; data = CUDA.pin(zeros(Float32, dx, np, length(querry_channel))))
    gpu_querry_struct = dataHolder(; data = gpu(querry_struct.data))
    CUDA.synchronize()

    task_vec = []
    other_task_vec = []
    for (i,(querry, response)) in enumerate(zip(querry_channel, response_channel))
        task = Threads.@spawn worker(querry, response; np, dx, calls)
        push!(task_vec, task)
        bind(querry, task)
        bind(response, task)
        errormonitor(task)

        task = Threads.@spawn try # exit gracefully on channel close
            while isopen(querry)
                push!(querry_struct, take!(querry), i)
            end        
        catch e
            isopen(querry) && rethrow(e)
        end
        push!(other_task_vec, task)
        errormonitor(task)
    end

    response_task = Threads.@spawn begin
        try
            while isopen(gpu_results_channel)
                (results, idxs) = take!(gpu_results_channel)
                results = cpu(results)
                for (i, idx) in enumerate(idxs)
                    dest_channel = response_channel[idx]
                    src_data = selectlastdim(results, i)
                    put!(dest_channel, src_data)
                end
            end
        catch e
            isopen(gpu_results_channel) && rethrow(e)
        end
    end
    errormonitor(response_task)

    t = 0
    @time while any(isopen, querry_channel) && any(isopen, response_channel)
        if !hasdata(querry_struct)
            sleep(0.001)
            continue
        end
        t += 1
        net_input, idxs = sendandreset!(gpu_querry_struct, querry_struct)
        results = net(net_input)
        put!(gpu_results_channel, (results, copy(idxs)))
    end
    close(gpu_results_channel)

    println(t)
end

struct noallocDense{F,M<:AbstractMatrix,B}
    weight::M
    bias::B
    σ::F  
end

m = net[1]
d = net[2]
y = net[1](x)

CUDA.@profile net(x)
CUDA.@profile m(x)

x = randn(Float32, 2, 500, 140) |> Cuda.pin
CUDA.@profile x |> gpu |> net |> cpu
CUDA.@time x |> gpu |> net |> cpu;

CUDA.@time gpu(x);

CUDA.@time sum(x);
v = vec(x)
CUDA.@time sum(v)

querry_channel = [Channel{AbstractArray}(1) for _ in 1:nworkers]
response_channel = [Channel{AbstractArray}(1) for _ in 1:nworkers]

tasks = []
for (i,(querry, response)) in enumerate(zip(querry_channel, response_channel))
    task = Threads.@spawn worker(querry, response)
    push!(tasks, task)
    bind(querry, task)
    bind(response, task)
    errormonitor(task)
end

CUDA.@time while !all(istaskdone, tasks)
    idxs = isready.(querry_channel)
    !any(idxs) && continue

    if count(idxs) == 1
        idx = findfirst(idxs)
        querries = take!(querry_channel[idx])
        querries_batched = reshape(querries, size(querries)..., 1)
        results = net(querries_batched |> gpu) |> cpu
        put!(response_channel[idx], results)    
    else
        querries = take!.(querry_channel[idxs])
        querries_batched = querries |> stack
        results = net(querries_batched |> gpu) |> cpu
        put!.(response_channel[idxs], eachcol(results))
    end
end




