struct Agent{}
    rs_hist::Vector{Float64}
    rb_hist::Vector{Float64}
    network_inputs::Matrix{T1}
    policy_targets::Matrix{T2}
    value_targets::Vector{T3}
    t_max::Int
    store_every::Int # if zero store only on t_max or episode end
end

struct ThreadedDataCollection{A, P, B, L<:Base.AbstractLock}
    agents::Vector{A}
    planners::Vector{P}
    querries::Vector{B}
    policy_response::Vector{Vector{Float32}}
    value_response::Vector{Float32}
    querry_locks::Vector{L}
    response_locks::Vector{L}
end

function handle_agent(S::ThreadedDataCollection, plan_idx::Int, new_root=false)
    planner = S.planners[plan_idx]
    agent = S.agents[plan_idx]

    if response_ready
        # backprop tree

        if search_done
            step(agent)
            new_root = true
        end    
    end

    if new_root
        # initialize tree
    end

    if response_ready || new_root
        add_querry
    end

    return nothing
end

