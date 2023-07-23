## Obtain U/t dependence of E_0 by Lanczos for 0 < U/t < 16.

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

function build_H(t, U)
    L = 6
    nelec = 6
    param = Hubbard1D.Param()
    Hubbard1D.initialize(param,t,U,L)
    v0 = zeros(Complex{Float64}, 2^(2*param.L))
    #
    Ndim = 0
    # println("basis")
    for k = 1:2^(2*param.L)
        if Hubbard1D.countbit(k-1,2*param.L) == nelec
            Ndim += 1
        # println(Ndim,"  ",string(k-1,base=2))
        end
    end
    # construct the hamiltonian matrix
    matrixH = zeros(Complex{Float64}, Ndim, Ndim)
    jcount = 0
    for k = 1:2^(2*param.L)
        if Hubbard1D.countbit(k-1,2*param.L) == nelec
            jcount +=1
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
    return matrixH
end

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


Ut = collect(0.1:0.1:8)
E0_values = zeros(length(Ut))

for (i, U_over_t) in enumerate(Ut)
    U = U_over_t
    t = 0.5
    println("U/t is ", U / t)
    H = build_H(t, U)  
    V, T = lanczos(H, 500)  
    eig = mkl.eigvals(T)
    E0 = eig[1]
    E0_values[i] = E0
end

p = scatter!(collect(0.1:0.1:8) / 0.5, E0_values, title = "\$E_0\$ and \$\\frac{U}{t}\$", xlabel = "\$\\frac{U}{t}\$", ylabel = "\$E_0\$", legend = false)

# savefig(p, "U_t_E_0.png")
savefig(p, "U_t_E_0_half.png")