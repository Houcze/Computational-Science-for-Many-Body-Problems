include("Hubbard1D.jl")
include("mkl.jl")

using .Hubbard1D
using .mkl
using LinearAlgebra
using Printf
using Plots
using LaTeXStrings
Plots.pythonplot()
default(fontfamily="Monaco")
Hubbard1D.parity(Int(0b1011)) # 0th up, 0th down, 1th down

# creation and annihilation operators
ell = 4
icomb = 2^ell
string(icomb, base=2)
i = Int(0b1101)                                 # 0th up, 1th up, 1th down 
println("i=",i)                                 # digit for "1101"
j,c1= Hubbard1D.Cop(1,i,1.0)                    # generate an down-spin electron at 0th site 
println(string(j, base=2),"  ",c1)
j,c1= Hubbard1D.Aop(2,i,1.0)                    # annihilate an up-spin electron at 1st site
string(j, base=2)

t = 1.0 # be careful about the boundary condition
U = 8.0
L = 6
nelec = 6
param = Hubbard1D.Param()
Hubbard1D.initialize(param,t,U,L)
v0 = zeros(Complex{Float64}, 2^(2*param.L))
#
Ndim = 0
println("basis")
for k = 1:2^(2*param.L)
  if Hubbard1D.countbit(k-1,2*param.L) == nelec
    global Ndim += 1
    println(Ndim,"  ",string(k-1,base=2))
  end
end
# construct the hamiltonian matrix
matrixH = zeros(Complex{Float64}, Ndim, Ndim)
jcount = 0
for k = 1:2^(2*param.L)
    if Hubbard1D.countbit(k-1,2*param.L) == nelec
        global jcount +=1
        v0[k] = 1.0
        v1 = zeros(Complex{Float64}, 2^(2*param.L))
        Hubbard1D.multiply(param,v0,v1)
        v0[k] = 0.0
        icount = 0
        for m = 1:2^(2*param.L)
            if Hubbard1D.countbit(m-1,2*param.L) == nelec
                icount += 1
                matrixH[icount,jcount] += v1[m]
            end
        end
    end
end

for i in 1:size(matrixH, 1)
    for j in 1:size(matrixH, 2)
        real_part = real(matrixH[i, j])
        imag_part = imag(matrixH[i, j])
        if abs(imag_part) < 1e-10  # add a tolerance for numerical precision
            @printf "%.2f\t" real_part
        else
            @printf "%.2f + %.2fim\t" real_part imag_part
        end
    end
    println()
end


println("==============================================================================================")
println("Obtain E_0 by Lapack:")
@time begin
eig = mkl.eigvals(real(matrixH))
end
wf = mkl.eigvecs(real(matrixH))
println("E_0=",eig[1])
wf[:,1]
println("2nd lowest eigenvalues=", eig[2])
println("3rd lowest eigenvalues=", eig[3])
# ccall((:diagonalize, "./lib/libjulia.so"), Cvoid, (Ptr{Cdouble}, Cint), matrixH, size(matrixH, 1))
println("==============================================================================================")

function lanczos(H, max_iter)
    n = size(H, 1)
    V = zeros(Float64, n, max_iter) 
    T = zeros(Float64, max_iter, max_iter)

    β = 0.0
    r = rand(Float64, n)
    for j = 1:max_iter
        v = r / norm(r)
        V[:,j] = v
        w = H * v
        α = real(dot(w, v))
        w -= α * v
        if j > 1
            w -= β * V[:,j-1]
        end
        β = norm(w)
        T[j,j] = α
        if j < max_iter
            T[j,j+1] = β
            T[j+1,j] = β
        end
        r = w
    end
    return V, T
end

function LANCZOS(H, max_iter)
    n = size(H, 1)
    V = zeros(ComplexF64, n, max_iter) 
    T = zeros(ComplexF64, max_iter, max_iter)
    eig1 = Float64[]

    β = 0.0
    r = rand(ComplexF64, n)
    for j = 1:max_iter
        v = r / norm(r)
        V[:,j] = v
        w = H * v
        α = real(dot(w, v))
        w -= α * v
        if j > 1
            w -= β * V[:,j-1]
        end
        β = norm(w)
        T[j,j] = α
        if j < max_iter
            T[j,j+1] = β
            T[j+1,j] = β
        end
        eig = mkl.eigvals(T)
        r = w
        push!(eig1, real(eig[1]))
    end
    return V, T, eig1
end
H = real(matrixH)
@time begin
V, T = lanczos(H, 500)
end
V, T, eig1 = LANCZOS(matrixH, 500)
eig = mkl.eigvals(T)
println("==============================================================================================")
println("LANCZOS: E_0=", eig[1])
println("==============================================================================================")
p = scatter!(1:500, eig1, title = L"E_0", xlabel = "Iteration Step", ylabel = L"E_0", label="\$E_0\$ at each Lanczos step", legend = :bottomright)
savefig(p, "ConvergenceE_0.png")

"""
open("T.txt", "w") do io
    for i in 1:size(T, 1)
        for j in 1:size(T, 2)
            real_part = real(T[i, j])
            imag_part = imag(T[i, j])
            if abs(imag_part) < 1e-10  
                write(io, @sprintf("%.2f\t", real_part))
            else
                write(io, @sprintf("%.2f + %.2fim\t", real_part, imag_part))
            end
        end
        write(io, "\n") 
    end
end
"""
H = real(matrixH)

function diagonal(μ)
    n = length(μ)
    diag_matrix = zeros(n, n)
    for i = 1:n
        diag_matrix[i, i] = μ[i]
    end
    return diag_matrix
end



"""
    Algorithm 5.1. The LOBPCG method I.
    Input: m starting vectors x^{(0)}_1, ... x^{(0)}_m, devices to compute: Ax, Bx, and Tx
        for a given vector x, and the vector inner product (x, y).
    1. Start: select x^{(0)}_j, and set p^{(0)}_j = 0, j = 1, ..., m.
    2. Iterate: For i = 0, . . . , Until Convergence Do:
    3. μ^{(i)}_j := (x^{(i)}_j, Bx^{(i)}_j)/(x^{(i)}_j, Ax^{(i)}_j), j=1, ..., m
    4. r_j := Bx^{(i)}_j - μ^{(i)}_j Ax^{(i)}_j, j=1, ..., m
    5. w^{(i)}_j := Tr_j, j=1, ..., m;
    6. Use the Rayleigh-Ritz method for the pencil B - μA
       on the trial subspace Span {w^{(i)}_1, ..., w^{(i)}_m, x^{(i)}_1, ..., x^{(i)}_m, p^{(i)}_1, ..., p^{(i)}_m}
    7. x^{(i+1)}_j := Σ_{k=1,...,m} α^{(i)}_k * w^{(i)}_k + τ^{(i)}_k * x^{(i)}_k + γ^{(i)}_k * p^{(i)}_k,
       (the jth Ritz vector corresponding to the jth Ritz value)
       j=1,...m;
    8. p^{(i+1)}_j := Σ_{k=1,...,m} α^{(i)}_k * w^{(i)}_k + γ^{(i)}_k * p^{(i)}_k
    9. EndDo
    Output: the approximations μ^{(k)}_j and x^{(k)}_j to the largest eigenvalues μ_j and corresponding eigenvectors, j=1,...m  
"""
function LOBCG(Ax, Bx, Tx, x0, maxiter=150, tol=1e-6)
    m = size(x0)[2]    
    X = x0
    P = zeros(Float64, size(X))
    W = zeros(Float64, size(X))
    μ = zeros(Float64, m)

    for i in 1:maxiter
        for j = 1:m
            x = X[:, j]
            μ[j] = dot(x, Bx(x)) / dot(x, Ax(x))
            r = Bx(x) - μ[j] * Ax(x)  
            W[:, j] = Tx(r)
        end
        # Build the matrices for the pencil B - μA in the subspace {w, x, p}
        Ψ = hcat(W, X, P)
        for j = 1:m    
            M = Ψ' * Bx(Ψ) - Ψ' * μ[j] * Ax(Ψ)

            eigvals, eigvecs = mkl.eigen(M)
            idx = partialsortperm(eigvals, j, rev=true)
            coeffs = eigvecs[:, idx]
 
            X[:, j] = Ψ * coeffs
            P[:, j] = hcat(Ψ[:, 1:m], Ψ[:, (2*m+1):end]) * vcat(coeffs[1:m], coeffs[(2*m+1):end])
        end

        converged = true
        for j in 1:m
            r = Bx(X[:, j]) - μ[j] * Ax(X[:, j])
            if norm(r) >= tol
                converged = false
                break
            end
        end
        if converged
            println("LOBCG converged in $i iterations")
            return μ, X
        end
              
    end
    error("LOBCG did not converge in $maxiter iterations")
end


Ax = x -> x
Bx = x -> -H * x
Tx = x -> x
x0 = rand(Float64, (size(H, 1), 3))
mu, x = LOBCG(Ax, Bx, Tx, x0)
println("==============================================================================================")
println("LOBCG: E_0=", -mu[1])
println("LOBCG: 2nd lowest eigenvalues=", -mu[2])
println("LOBCG: 3rd lowest eigenvalues=", -mu[3])
println("==============================================================================================")

