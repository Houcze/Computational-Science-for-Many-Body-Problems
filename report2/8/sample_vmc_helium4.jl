include("Helium4.jl")

using .Helium4
using Random
using LaTeXStrings
using Plots
using DelimitedFiles
Plots.pythonplot()

a1 = 2.6 # variational parameter
a2 = 5.0 # variational parameter
L = 11.2 # linear length of the cubic cell (periodic boundary condition) (ang)
N = 32   # number of atoms in the cubic cell

param = Helium4.Param()
Helium4.initialize(param, a1, a2, L, N)
Helium4.initialize_config(param)

Nwup = 3200         # number of warming up steps
Nsample = 12800     # number of MC samples
Random.seed!(10)    # set a seed for the psuedo random number generator

# preparation for the radial two-point distribution function
Ndr = 64
Nrdr = zeros(Float64, Ndr)
gofr = zeros(Float64, Ndr)
vecr = zeros(Float64, Ndr)

# warming up
config_old = zeros(Float64, 3, param.N)
dist_old = zeros(Float64, param.N, param.N)
for i in 1:Nwup
    for j in 1:param.N
        config_old .= param.config
        dist_old .= param.dist
        iupdate = Int(floor(rand(Float64)*param.N)) + 1
        if iupdate > param.N
            println("error!", iupdate)
        end
        Helium4.update_config(param,iupdate)
        Helium4.update_dist(param,iupdate)
        lnPNt = 0.0
        lnPNi = 0.0
        for m in 1:param.N
            if m ≠ iupdate
                lnPNt -= 2.0*Helium4.func_u(param,param.dist[m,iupdate])
                lnPNi -= 2.0*Helium4.func_u(param,dist_old[m,iupdate])
                if (param.dist[m,iupdate]- dist_old[m,iupdate]) == 0.0
                   println("dist is not")
                end
            end
        end
        ratio = exp(lnPNt - lnPNi)
        if ratio <= rand(Float64)
            param.config[:,:] = config_old[:,:]
            param.dist[:,:] = dist_old[:,:]
        end
    end
    if i % 100 == 0
        println("Wup",i," ",param.config[1,1])
    end
end

# MC sampling
for i in 1:Nsample
    for j in 1:param.N
        config_old .= param.config
        dist_old .= param.dist
        iupdate = Int(floor(rand(Float64)*param.N)) + 1
        Helium4.update_config(param,iupdate)
        Helium4.update_dist(param,iupdate)
        lnPNt = 0.0
        lnPNi = 0.0
        for m in 1:param.N
            if m ≠ iupdate
                lnPNt -= 2.0*Helium4.func_u(param,param.dist[m,iupdate])
                lnPNi -= 2.0*Helium4.func_u(param,dist_old[m,iupdate])
            end
        end
        ratio = exp(lnPNt - lnPNi)
        if ratio <= rand(Float64)
            param.config[:,:] = config_old[:,:]
            param.dist[:,:] = dist_old[:,:]
        end
    end
    if i % 100 == 0
        println("MCstep",i," ",param.config[1,1])
    end
    Helium4.accum_Nrdr(param,Ndr,Nrdr)
end

Helium4.calc_gofr(param,Nsample,Ndr,Nrdr,gofr,vecr)

plot(vecr,gofr,xlabel=L"r",ylabel=L"g(r)")
savefig("sample_vmc_helium4.png")

writedlm("vecr.txt", vecr)
writedlm("gofr.txt", gofr)

