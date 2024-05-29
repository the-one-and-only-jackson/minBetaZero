using Flux, CUDA

include("transformer.jl")

function local_SetTransformer(params::TransformerParams; induced=true)
    embed = [Dense(params.d_in=>params.d)]
    if induced
        encoder = [InducedAttBlock(params.k_enc, params) for _ in 1:params.n_enc]
    else
        encoder = [SelfAttBlock(params) for _ in 1:params.n_enc]
    end
    decoder = [LearnedQuerriesBlock(params.k_dec, params), SelfAttBlock(params)]
    output = Any[]
    params.prenorm && pushfirst!(output, params.norm(params.d))
    Chain(embed..., encoder..., decoder..., output...)
end

lightdark_st() = Chain(
    x->reshape(x, size(x,1), size(x,2), :),
    local_SetTransformer(TransformerParams(; 
        d_in = 1,
        d = 64, # d = dout because this is a shared layer
        dropout = 0.1,
        n_enc = 2,
        n_dec = 1,
        k_enc = 4,
        k_dec = 1,
    )),
    x->dropdims(x; dims=2)
)

function worker_job(querry_channel, response_channel)
    output = 0f0
    for _ in 1:10
        x = randn(Float32, 1, 100)
        put!(querry_channel, x)
        y = take!(response_channel)
        output += sum(y)
    end
    output
end

function test(net)
    N_tasks = 500
    worker_querries = [Channel{Array{Float32}}(1) for _ in 1:N_tasks]
    master_responses = [Channel{Array{Float32}}(1) for _ in 1:N_tasks]

    gpu_net = net |> gpu

    @time begin
        futures = [Threads.@spawn worker_job(q,r) for (q,r) in zip(worker_querries, master_responses)]

        while !all(istaskdone, futures)
            idxs = isready.(worker_querries)
            iszero(count(idxs)) && continue

            data = stack(take!, worker_querries[idxs])
            results = data |> gpu |> gpu_net |> cpu
            put!.(master_responses[idxs], eachcol(results))
        end

        fetch.(futures)
    end
end

net = lightdark_st()
test(net);







