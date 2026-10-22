# The Julia binding's own cost per solve: the median Julia wall time of
# `ApxChol.solve!` minus the `solve_seconds` apxchol measures inside the same
# call, and the Julia bytes it allocates. A diagnostic of the wrapper, not a
# solver benchmark: run it alone on an otherwise idle machine.
#
#     julia --project=. bench/overhead.jl
using ApxChol, LinearAlgebra, SparseArrays

function grid_laplacian(rows, cols; shift)
    id(r, c) = (r - 1) * cols + c
    I = Int[]; J = Int[]
    edge(u, v) = (append!(I, (u, v)); append!(J, (v, u)))
    for r in 1:rows, c in 1:cols
        r < rows && edge(id(r, c), id(r + 1, c))
        c < cols && edge(id(r, c), id(r, c + 1))
    end
    A = sparse(I, J, -1.0, rows * cols, rows * cols)
    return A - spdiagm(0 => vec(sum(A; dims=2)) .- shift)
end

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

function measure(S, x, b, repetitions)
    wall = Vector{Float64}(undef, repetitions)
    native = Vector{Float64}(undef, repetitions)
    for i in 1:repetitions
        t = time_ns()
        r = ApxChol.solve!(x, S, b)
        wall[i] = (time_ns() - t) / 1e9
        native[i] = r.solve_seconds
    end
    return wall, native
end

println("ApxChol ", pkgversion(ApxChol), ", apxchol ", ApxChol.version(), ", Julia ", VERSION,
        ", ", Sys.cpu_info()[1].model, ", ", Threads.nthreads(), " Julia threads")
println(join(sort!(["$k=$v" for (k, v) in ApxChol.build_info()]), ", "))
println("rows     repetitions  wall median (µs)  apxchol median (µs)  difference (µs)  Julia bytes")
for side in (10, 100, 300)
    A = grid_laplacian(side, side; shift=0.01)
    n = size(A, 1)
    S = ApxCholSolver(A; backend=:cpu, threads=1)
    b = normalize!(collect(range(-1.0, 1.0; length=n)))
    x = zeros(n)
    repetitions = side == 300 ? 50 : 500
    measure(S, x, b, 5)                                  # warm up
    bytes = @allocated ApxChol.solve!(x, S, b)
    wall, native = measure(S, x, b, repetitions)
    w, c = 1e6 * median(wall), 1e6 * median(native)
    println(rpad(n, 9), rpad(repetitions, 13), rpad(round(w; digits=1), 18),
            rpad(round(c; digits=1), 21), rpad(round(w - c; digits=2), 17), bytes)
    close(S)
end
