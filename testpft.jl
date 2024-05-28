using ParticleFilterTrees
using POMDPs, POMDPTools, QuickPOMDPs
using Distributions, Statistics, Random
using ParticleFilters
using ProgressMeter
using Plots
using MCTS, QMDP 

# note: pft works

const R = 60
const LIGHT_LOC = 10

LightDarkPOMDP() = QuickPOMDP(
    states = -R:R+1, # r+1 is a terminal state
    stateindex = s -> s + R + 1,
    actions = [-10, -1, 0, 1, 10],
    discount = 0.95,
    isterminal = s::Int -> s==R::Int+1,
    obstype = Float64,

    transition = function (s::Int, a::Int)
        if a == 0
            return Deterministic{Int}(R::Int+1)
        else
            return Deterministic{Int}(clamp(s+a, -R::Int, R::Int))
        end
    end,

    observation = (s, a, sp) -> Normal(sp, abs(sp - LIGHT_LOC::Int) + 1e-3),

    reward = function (s, a)
        if iszero(a)
            return iszero(s) ? 100.0 : -100.0
        else
            return -1.0
        end
    end,

    initialstate = POMDPTools.Uniform(div(-R::Int,2):div(R::Int,2))
)

function mysimulator(pomdp, planner; updater_p = 10_000, max_steps = 50)
    up = BootstrapFilter(pomdp, updater_p)
    b = initialize_belief(up, initialstate(pomdp))
    s = rand(initialstate(pomdp))
    disc = 1.0
    r_total = 0.0
    step = 1
    while disc > eps() && !isterminal(pomdp, s) && step <= max_steps
        a = action(planner, b)
        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
        r_total += disc*r
        s = sp
        bp = update(up, b, a, o)
        b = bp
        disc *= discount(pomdp)
        step += 1
    end
    r_total    
end

function mysimulator2(pomdp, planner; updater_p = 10_000, max_steps = 50)
    up = BootstrapFilter(pomdp, updater_p)
    b = initialize_belief(up, initialstate(pomdp))
    disc = 1.0
    r_total = 0.0
    step = 1
    while disc > eps() && !all(isterminal(pomdp, s) for s in particles(b)) && step <= max_steps
        a = action(planner, b)
        o = @gen(:o)(pomdp, rand(b), a)
        r_total += disc * mean(@gen(:r)(pomdp,s,a) for s in particles(b))
        b = update(up, b, a, o)
        disc *= discount(pomdp)
        step += 1
    end
    r_total    
end

pomdp = LightDarkPOMDP()

PO_VE = ParticleFilterTrees.PORollout(QMDPSolver(); n_rollouts=1)
sol = PFTDPWSolver(
    tree_queries=1000,
    max_depth=28,
    k_o = 24.,
    criterion = ParticleFilterTrees.MaxPoly(95., 0.39),
    resample = false,
    value_estimator = PO_VE,
    check_repeat_obs = true
)
planner = solve(sol, pomdp)
mysimulator(pomdp, planner; updater_p=500) # 60 +- 20

ret = [mysimulator2(pomdp, planner; updater_p=500) for _ in 1:10]
mean(ret)
std(ret)/sqrt(length(ret))
# resample
# s: 55 +- 1.5
# b: 55 +- 2.3

# Pkg.develop(PackageSpec(url=joinpath(@__DIR__,"lib","ParticleFilterTrees")))

# tree seach is fine!

using minBetaZero

pomdp = LightDarkPOMDP()
bmdp = ParticleBeliefMDP(pomdp)
sol = PFTDPWSolver(
    tree_queries=1000,
    max_depth=28,
    k_o = 24.,
    criterion = ParticleFilterTrees.MaxPoly(95., 0.39),
    resample = false,
    value_estimator = ParticleFilterTrees.PORollout(QMDPSolver(); n_rollouts=1),
    check_repeat_obs = true
)
planner = solve(sol, pomdp)

ret = [mysimulator3(bmdp, planner; updater_p=500) for _ in 1:100] # 60 +- 20
mean(ret)
std(ret)/sqrt(length(ret))


function mysimulator3(bmdp::ParticleBeliefMDP, planner; updater_p = 10_000, max_steps = 50)
    b = rand(initialstate(bmdp))
    disc = 1.0
    r_total = 0.0
    step = 1
    while disc > eps() && !isterminal(bmdp, b) && step <= max_steps
        a = action(planner, b)
        a_idx = actionindex(bmdp.pomdp, a)
        b, r = @gen(:sp, :r)(bmdp, b, a_idx)
        r_total += disc * r
        disc *= discount(pomdp)
        step += 1
    end
    r_total    
end


