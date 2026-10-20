using ApxChol
using LinearAlgebra
using Printf
using SparseArrays

function matpower_matrix(source::String, name::String)
    block = match(Regex("mpc\\.$name\\s*=\\s*\\[(.*?)\\];", "s"), source)
    isnothing(block) && error("missing mpc.$name matrix")

    rows = Vector{Vector{Float64}}()
    for raw_line in eachline(IOBuffer(only(block.captures)))
        line = strip(first(split(raw_line, '%'; limit=2)))
        isempty(line) && continue
        fields = split(replace(line, ';' => ' '))
        isempty(fields) || push!(rows, parse.(Float64, fields))
    end
    isempty(rows) && error("empty mpc.$name matrix")
    width = length(first(rows))
    all(length(row) == width for row in rows) || error("ragged mpc.$name matrix")
    return reduce(vcat, permutedims.(rows))
end

function case_laplacian(path::String)
    source = read(path, String)
    buses = matpower_matrix(source, "bus")
    branches = matpower_matrix(source, "branch")
    size(buses, 1) == 9_241 || error("expected 9,241 buses, found $(size(buses, 1))")
    size(branches, 2) >= 11 || error("branch matrix has too few columns")

    bus_index = Dict(round(Int, buses[row, 1]) => row for row in axes(buses, 1))
    row_indices = Int[]
    column_indices = Int[]
    values = Float64[]
    for branch in eachrow(branches)
        iszero(branch[11]) && continue
        from = bus_index[round(Int, branch[1])]
        to = bus_index[round(Int, branch[2])]
        reactance = branch[4]
        iszero(reactance) && error("active branch has zero reactance")
        weight = inv(abs(reactance))
        append!(row_indices, (from, to, from, to))
        append!(column_indices, (from, to, to, from))
        append!(values, (weight, weight, -weight, -weight))
    end
    return sparse(row_indices, column_indices, values, size(buses, 1), size(buses, 1))
end

function main(path::String)
    matrix = case_laplacian(path)
    rhs = [isodd(i) ? 1.0 : -1.0 for i in axes(matrix, 1)]
    rhs .-= sum(rhs) / length(rhs)

    # The settings this regression has always used (not the library defaults).
    for threads in (1, 2, 4, 8)
        solver = ApxCholSolver(matrix; rtol=1e-8, maxiter=1000, stagnation_window=0,
                               seed=7, threads=threads)
        result = ApxChol.solve(solver, rhs)
        residual = norm(matrix * result.x - rhs) / norm(rhs)
        result.converged || error("$threads-thread solve did not converge")
        residual <= 1e-8 || error("$threads-thread residual $residual exceeds 1e-8")
        @printf(
            "threads=%d iterations=%d residual=%.3e setup=%.6f solve=%.6f\n",
            threads,
            result.iterations,
            residual,
            ApxChol.stats(solver).setup_seconds,
            result.solve_seconds,
        )
        close(solver)
    end
end

length(ARGS) == 1 || error("usage: julia --project=. test/regression_case9241pegase.jl CASE")
main(only(ARGS))
