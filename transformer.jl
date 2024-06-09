
@kwdef struct TransformerParams
    d_in = 2
    d_out = 24
    d = 128
    dhidden = 4*d
    prenorm = true
    activation = gelu
    norm = RMSNorm
    init = Flux.orthogonal
    nheads = 4
    dropout = 0.1
    n_enc = 2
    n_dec = 1
    k_enc = 4
    k_dec = 4
end

struct RMSNorm{S, B}
    scale::S
    bias::B
end
RMSNorm(d; bias=true) = RMSNorm(ones(Float32,d), bias ? zeros(Float32,d) : false)
(block::RMSNorm)(x) = block.scale .* x .* sqrt.(size(x,1) ./ (sum(abs2, x; dims=1) .+ eps(eltype(x)))) .+ block.bias
Flux.@layer :expand RMSNorm

function Flux.ChainRulesCore.rrule(block::T, x) where {T<:RMSNorm}
    d = size(x,1)

    dot_xx = sum(abs2, x; dims=1) .+ eps(eltype(x))
    inv_rms = sqrt.(d ./ dot_xx)
    out_1 = x .* inv_rms
    out_2 = block.scale .* out_1 .+ block.bias

    function get_RMSNorm_pullback(Ȳ)
        scale_bar = dropdims(mapreduce(*, +, reshape(Ȳ, d, :), reshape(out_1, d, :); dims=2); dims=2)
        bias_bar = dropdims(sum(reshape(Ȳ, d, :); dims=2); dims=2)
        struct_tangent = Flux.ChainRulesCore.Tangent{T}(; scale=scale_bar, bias=bias_bar)

        Y_bar = block.scale .* Ȳ
        dot_xy = mapreduce(*, +, x, Y_bar; dims=1)
        x_bar = inv_rms .* (Y_bar .- dot_xy ./ dot_xx .* x)
        return struct_tangent, x_bar
    end

    return out_2, get_RMSNorm_pullback
end

function FeedForwardResBlock(params::TransformerParams)
    if params.prenorm
        ff_net = Chain(
            params.norm(params.d; bias=false), 
            Dense(params.d=>params.dhidden, params.activation),
            Dense(params.dhidden=>params.d),
            Dropout(params.dropout)
        )
        net = SkipConnection(ff_net, +)
    else
        ff_net = Chain(
            Dense(params.d=>params.dhidden, params.activation),
            Dense(params.dhidden=>params.d; bias=false),
            Dropout(params.dropout)
        )
        net = Chain(SkipConnection(ff_net, +), params.norm(params.d))
    end
    return net
end

struct CrossAttResBlock{T1,T2,T3}
    mha::T1
    norm::T2
    KV_prenorm::T3
    prenorm::Bool
end
function CrossAttResBlock(params::TransformerParams)
    CrossAttResBlock(
        MultiHeadAttention(params.d; params.nheads, dropout_prob=params.dropout), 
        params.norm(params.d), 
        params.norm(params.d),
        params.prenorm
    )
end
function (block::CrossAttResBlock)(Q, KV)
    Qin = block.prenorm ? block.norm(Q) : Q
    KVin = block.prenorm ? block.KV_prenorm(KV) : KV
    res_out = Q + block.mha(Qin, KVin)[1]
    return block.prenorm ? res_out : block.norm(res_out)
end
function (block::CrossAttResBlock)(Q)
    Qin = block.prenorm ? block.norm(Q) : Q
    res_out = Q + block.mha(Qin)[1]
    return block.prenorm ? res_out : block.norm(res_out)
end
Flux.@layer :expand CrossAttResBlock

struct CrossAttBlock{T1<:CrossAttResBlock, T2}
    att_block::T1
    ff_block::T2
end
function CrossAttBlock(params::TransformerParams)
    att_block = CrossAttResBlock(params)
    ff_block = FeedForwardResBlock(params)
    return CrossAttBlock(att_block, ff_block)
end
(block::CrossAttBlock)(args...) = block.att_block(args...) |> block.ff_block
Flux.@layer :expand CrossAttBlock

SelfAttBlock(params) = CrossAttBlock(params)

struct LearnedQuerriesBlock{T1<:AbstractArray, T2<:CrossAttBlock}
    Q::T1
    att_block::T2
end
function LearnedQuerriesBlock(n, params::TransformerParams)
    Q = params.init(params.d, n)
    att_block = CrossAttBlock(params)
    LearnedQuerriesBlock(Q, att_block)
end
(block::LearnedQuerriesBlock)(X) = block.att_block(repeat(block.Q,1,1,size(X)[3:end]...), X)
Flux.@layer :expand LearnedQuerriesBlock

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




# # NEW

# struct SetTransformer
#     encoder_blocks
#     decoder_blocks
#     Q
#     embedding
#     output
# end
# function SetTransformer(params::TransformerParams)
#     encoder_blocks = [EncoderBlock(params) for _ in 1:params.n_enc]
#     decoder_blocks = [EncoderBlock(params) for _ in 1:params.n_dec]
#     Q = Flux.glorot_normal(params.d, params.k_dec)
#     embedding = Dense(params.d_in=>params.d)
#     output = Chain(params.norm(params.d; bias=false), Dense(params.d=>params.d_out÷params.k_dec))
# end
# function (st::SetTransformer)(x)
#     st.embedding(x)
# end
# Flux.@layer :expand SetTransformer

# struct DecoderBlock{CA, SA, F, CN, SN, FN}
#     cross_attention::CA
#     self_attention::SA
#     ff::F
#     cross_norm::CN
#     self_norm::SN
#     ff_norm::FN
# end
# function DecoderBlock(params::TransformerParams)
#     cross_attention = MultiHeadAttention(params.d; params.nheads, dropout_prob=params.attention_dropout)
#     self_attention  = MultiHeadAttention(params.d; params.nheads, dropout_prob=params.attention_dropout)
#     ff = Chain(Dense(params.d=>params.dhidden, params.activation), Dense(params.dhidden=>params.d))
#     cross_norm = params.norm(params.d; bias=true)
#     self_norm  = params.norm(params.d; bias=true)
#     ff_norm    = params.norm(params.d; bias=false)
#     DecoderBlock(cross_attention, self_attention, ff, cross_norm, self_norm, ff_norm)
# end 
# function (block::DecoderBlock)(Q,KV)
#     # Assumes KV are already normed
#     Y1 = Q + block.cross_attention(block.cross_norm(Q), KV)
#     Y2 = Y1 + block.self_attention(block.self_norm(Y1))
#     Y3 = Y2 + ff(block.ff_norm(Y2))
#     return Y3
# end
# Flux.@layer :expand DecoderBlock

# struct EncoderBlock{A,F}
#     self_attention::A
#     ff::F
#     self_norm = params.norm(params.d; bias=true)
#     ff_norm   = params.norm(params.d; bias=false)
# end
# function EncoderBlock(params::TransformerParams)
#     self_attention = if params.induced
        
#     else
#         MultiHeadAttention(params.d; params.nheads, dropout_prob=params.attention_dropout)
#     end
#     ff = FeedForwardBlock(params)
#     self_norm  = params.norm(params.d; bias=true)
#     ff_norm    = params.norm(params.d; bias=false)
#     DecoderBlock(cross_attention, self_attention, ff, cross_norm, self_norm, ff_norm)
# end
# function (block::EncoderBlock)(X)
#     Y1 = X + block.self_attention(block.self_norm(X))
#     Y2 = Y1 + block.ff(block.ff_norm(Y1))
#     return Y2
# end
# Flux.@layer :expand EncoderBlock

# # need to figure out how to do dropout, their implementation uses inv(1-p) which is bad
# # NeuralAttentionlib.multihead_qkv_attention()




