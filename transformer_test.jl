using Flux, CUDA, NeuralAttentionlib, Random
using NeuralAttentionlib: dropoutF


struct InducedAttBlock{T1<:LearnedQuerriesBlock, T2<:CrossAttBlock}
    gen_kv::T1
    att_block::T2
end
function InducedAttBlock(n, params::TransformerParams)
    InducedAttBlock(LearnedQuerriesBlock(n, params), CrossAttBlock(params))
end
(block::InducedAttBlock)(X) = block.att_block(X, block.gen_kv(X))
Flux.@layer :expand InducedAttBlock

function SetTransformer(params::TransformerParams; induced=true)
    embed = [Dense(params.d_in=>params.d)]
    if induced
        encoder = [InducedAttBlock(params.k_enc, params) for _ in 1:params.n_enc]
    else
        encoder = [SelfAttBlock(params) for _ in 1:params.n_enc]
    end
    decoder = [LearnedQuerriesBlock(params.k_dec, params), SelfAttBlock(params)]
    output = Any[Dense(params.d=>params.d_out÷params.k_dec)]
    params.prenorm && pushfirst!(output, params.norm(params.d))
    Chain(embed..., encoder..., decoder..., output...)
end


head = 4
input_dims = 7
head_dims = 5
output_dims = 8


rng1 = Xoshiro()
dp1 = dropoutF(; rng = rng1, p = 0.5)
x = randn(Float32, input_dims, 3, 2)

device = cpu
mha = device(mha)
x = device(x)

@test mha(x,x,x) ≈ atten(mha, x; p = dp1)

head = 4
input_dims = 7
head_dims = 5
output_dims = 8

function atten(mha, x, mask=nothing, return_score = false; p = nothing)
    q = mha.iqproj(x)
    k = mha.ikproj(x)
    v = mha.ivproj(x)
    if return_score
        a, score = NeuralAttentionlib.multihead_qkv_attention(score_returning, 4, q, k, v, mask, p)
        return mha.oproj(a), score
    else
        a = NeuralAttentionlib.multihead_qkv_attention(4, q, k, v, mask, p)
        return mha.oproj(a)
    end
end

multihead_qkv_attention(4, q, k, v, mask, p)


function flux_test(;
    d = 128,
    nheads = 4,
    dropout_prob = 0.1,
    device = cpu,
    )
    
    mha = MultiHeadAttention(d; nheads, dropout_prob) |> device
    
    x = randn(Float32, d, d, 32) |> device
    
    CUDA.@time mha(x)[1];
end

function lib_test(;
    d = 128,
    nheads = 4,
    dropout_prob = 0.1,
    device = cpu,
    )

    WQ = randn(Float32, d, d) |> device
    WK = randn(Float32, d, d) |> device
    WV = randn(Float32, d, d) |> device
    WO = randn(Float32, d, d) |> device

    mask = nothing
    p = dropoutF(; p = dropout_prob)
        
    x = randn(Float32, d, d, 32) |> device

    CUDA.@time begin
        q = batched_mul(WQ, x)
        k = batched_mul(WK, x)
        v = batched_mul(WV, x)
        y = NeuralAttentionlib.multihead_qkv_attention(nheads, q, k, v, mask, p)
        z = batched_mul(WO, y)
    end
end

flux_test(; device=gpu);
lib_test(; device=gpu);

function test()
    x = randn(Float32, 128, 128, 32)
    w = randn(Float32, 128, 128)

    @time y1 = batched_mul(w, x)
    @time y2 = mybatchedmul(w, x)
    y1 ≈ y2
end
function mybatchedmul(w, x)
    x2d = reshape(x, size(x,1), :)
    wx2d = w * x2d
    return reshape(wx2d, size(w,1), size(x)[2:end]...)
end

test()


struct CrossAttention{Q<:Union{AbstractMatrix, Nothing}, K<:AbstractMatrix, V<:AbstractMatrix, O<:AbstractMatrix, DF<:dropoutF, MK}
    WQ::Q
    WK::K
    WV::V
    WO::O
    df::DF
    nheads::Int
    mask::MK
end
CrossAttention(; d, nheads, init, p) = CrossAttention(
    init(d,d),
    init(d,d),
    init(d,d),
    init(d,d),
    dropoutF(; p),
    nheads,
    nothing
)

(ca::CrossAttention)(x) = ca(x, x, x)
(ca::CrossAttention)(q, kv) = ca(q, kv, kv)
function (ca::CrossAttention)(q, k, v)
    wq = isnothing(ca.WQ) ? q : batched_mul(ca.WQ, q)
    wk = batched_mul(ca.WK, k)
    wv = batched_mul(ca.WV, v)
    o = NeuralAttentionlib.multihead_qkv_attention(ca.nheads, wq, wk, wv, ca.mask, ca.df)
    return batched_mul(ca.WO, o)
end

function test()
    ca = CrossAttention(; d=128, nheads=4, init=(sz...)->randn(Float32,sz...), p=0.1)
    x = randn(Float32, 128, 128, 32)
    @time ca(x);
end

ca = CrossAttention(; d=128, nheads=4, init=(sz...)->randn(Float32,sz...), p=0.1)
x = randn(Float32, 128, 128, 32)
q = randn(Float32, 128, 1, 32)
@time ca(q, x)

@time repeat(q, 1, 1, 3);

struct FeedForward{
    W1<:AbstractMatrix,
    W2<:AbstractMatrix,
    B1<:Union{AbstractVector, Bool},
    B2<:Union{AbstractVector, Bool},
    D,
    F
    }
    w1::W1
    w2::W2
    b1::B2
    b2::B1
    d::D
    act::F
end
function FeedForward((in,(hidden,out))::Pair{Int,Pair{Int,Int}}, act_fun=relu; init, b1=false, b2=false, p=nothing)
    FeedForward(
        init(hidden, in),
        init(out, hidden),
        b1 ? init(hidden) : false,
        b2 ? init(out) : false,
        isnothing(p) ? identity : Flux.Dropout(p1),
        act_fun
    )
end
function (ff::FeedForward)(x)
    y1 = batched_mul(ff.w1, x) .+ ff.b1
    y2 = ff.act.(y1)
    y3 = batched_mul(ff.w2, y2) .+ ff.b2
    return ff.d2(y3)
end

struct InducedEncoder{
    M1<:CrossAttention, M2<:CrossAttention, 
    F1<:FeedForward, F2<:FeedForward, 
    L1<:LayerNorm, L2<:LayerNorm, L3<:LayerNorm, L4<:LayerNorm,
    Q<:AbstractMatrix
    }
    mha_1::M1
    mha_2::M2
    ff_1::F1
    ff_2::F2
    ln_1::L1
    ln_2::L2
    ln_3::L3
    ln_4::L4
    q::Q
end

function test()
    d = Dense(128=>128; bias=false)
    x = randn(Float32, 128, 128, 32)
    @time d(x)
end
function test()
    w = randn(Float32, 128, 128)
    x = randn(Float32, 128)
    @time batched_mul(w, x)
end
test();


