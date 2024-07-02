struct CGF{W<:AbstractMatrix}
    weight::W
end
Flux.@layer CGF

function CGF((dx,nv)::Pair{<:Integer, <:Integer}; init=randn, T=Float32)
    CGF(init(T, (dx, nv)))
end

function (c::CGF)(x::AbstractArray{<:Any, 2})
    x3d = reshape(x, size(x,1), size(x,2), 1)
    c2d = c(x3d)
    return dropdims(c2d, dims=2)
end

function (c::CGF)(x::AbstractArray{Float32, 3})
    (dx, nx, _) = size(x)
    wx = batched_mul(c.weight', x) ./ Float32(sqrt(dx))
    dropdims(logsumexp(wx; dims=2); dims=2) .- Float32(log(nx))
end


struct RMSNorm{S, B}
    scale::S
    bias::B
end
Flux.@layer :expand RMSNorm

RMSNorm(d; bias=true) = RMSNorm(ones(Float32,d), bias ? zeros(Float32,d) : false)

function (block::RMSNorm)(x)
    dx = size(x, 1)
    sx = sum(abs2, x; dims=1) .+ eps(eltype(x))
    @. block.scale * x * sqrt(dx / sx) + block.bias
end
