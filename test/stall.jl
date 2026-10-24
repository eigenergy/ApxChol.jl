# Run by test/safety.jl as `julia -t 2,1 test/stall.jl`. Prints how long one
# factorization, solve and factor export took on the main thread, the longest
# pause an allocating loop on another thread saw meanwhile, and the longest
# time a collection waited for threads to reach a safepoint. A collection that
# loop triggers waits until every thread reaches one; a native call that is
# not GC-safe holds it there until the call returns.
using ApxChol
using LinearAlgebra
using SparseArrays

function grid(k)
    T = spdiagm(-1 => fill(-1.0, k - 1), 1 => fill(-1.0, k - 1))
    A = kron(T, sparse(1.0I, k, k)) + kron(sparse(1.0I, k, k), T)
    return A - spdiagm(0 => vec(sum(A; dims=2)))
end

function worst_stall(f)
    stop = Threads.Atomic{Bool}(false)
    main = Threads.threadid()
    # The loop never yields, so it must not share the main thread.
    loop = Threads.@spawn :default begin
        Threads.threadid() == main && error("the allocation loop shares the main thread")
        worst, last = 0.0, time()
        while !stop[]
            v = Vector{Float64}(undef, 1 << 17)
            v[1] = 1.0
            now = time()
            worst, last = max(worst, now - last), now
        end
        worst
    end
    sleep(0.2)
    seconds = @elapsed f()
    stop[] = true
    return seconds, fetch(loop)
end

A = grid(500)
b = [sin(0.37i) for i in 1:size(A, 1)]
b .-= sum(b) / length(b)
function long_calls()
    S = ApxCholSolver(A; backend=:cpu, threads=1, keep_factor=true)
    ApxChol.solve(S, b; rtol=1e-300, maxiter=1000)
    ApxChol.factor(S)
    close(S)
end
long_calls()                                     # compile first
seconds, stall = worst_stall(long_calls)
println(seconds, " ", stall, " ", Base.gc_num().max_time_to_safepoint / 1e9)
