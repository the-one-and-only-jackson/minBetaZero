plot_smoothing(y; kwargs...) = plot_smoothing(1:length(y), y; kwargs...)
plot_smoothing(x, y; kwargs...) = plot_smoothing!(plot(), x, y; kwargs...)
plot_smoothing!(p, y; kwargs...) = plot_smoothing!(p, 1:length(y), y; kwargs...)
function plot_smoothing!(p, x, y; nx=250, k=1, kwargs...)
    x_new_maybe = range(minimum(x), stop=maximum(x), length=nx)
    x_new = Float64[]
    y_new = Float64[]
    for i in eachindex(x_new_maybe)
        x1 = x_new_maybe[max(1, i-k)]
        x2 = x_new_maybe[min(nx, i+k)]
        idxs = findall(x1 .<= x .<= x2)
        if !isempty(idxs)
            push!(x_new, mean(x[idxs]))
            push!(y_new, mean(y[idxs]))
        end
    end
    plot!(p, x_new, y_new; kwargs...)
end
