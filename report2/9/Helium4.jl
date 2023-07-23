module Helium4

# physical constant
    hbar = 1.054571e-34 # kg m^2 / s
    kB = 1.380649e-23 # J / K
  # mass of helium 4 atom
    massHe4 = 4.00260325415 * 1.0e-3 / 6.022140e+23 # kg
    mass = kB * massHe4 * 1.0e-20 / hbar^2 # K^-1 angstrom^-2
  # parameters for Lenard-Jones 6-12 potential
    epsLJ = 1.41102e-22 / kB # K
    sigmaLJ = 2.556 # angstrom

# parameters and valuables
    mutable struct Param
        a1::Float64
        a2::Float64
        L::Float64
        N::Int
        rcut::Float64
        config::Array{Float64,2} # (3,N)
        dist::Array{Float64,2} #  (N,N)
        Param() = new()
    end

# initialization
    function initialize(param::Param, a1, a2, L, N)
        param.a1 = a1
        param.a2 = a2
        param.L = L
        param.N = N
        param.rcut = 0.5*a1
        param.config = zeros(Float64,3,N) # column-major
        param.dist = zeros(Float64,N,N) # column-major
        return 0
    end

    function initialize_config(param::Param)
        for j in 1:param.N
            for i in 1:3
                param.config[i,j] = param.L * rand(Float64)
            end
        end
        for iupdate in 1:param.N
            update_dist(param, iupdate)
        end
    end

# update the configuration of the atoms
    function update_config(param::Param, iupdate)
        d = 0.05*param.L
        for i in 1:3
            param.config[i,iupdate] += 2.0*d*(rand(Float64)-0.5)
            param.config[i,iupdate] = mod(param.config[i,iupdate], param.L)
        end
    end

    function update_dist(param::Param, iupdate)
        for i in 1:param.N
            tmp_dist = dist_periodic(param,param.config[:,i],param.config[:,iupdate])
            if tmp_dist > param.rcut
                param.dist[i,iupdate] = tmp_dist
                param.dist[iupdate,i] = tmp_dist
            else
                param.dist[i,iupdate] = param.rcut
                param.dist[iupdate,i] = param.rcut
            end
        end
    end

    function dist_periodic(param::Param, r1, r2)
        dist_old = sqrt(3.0)*param.L
        for iz in -1:1
            for iy in -1:1
                for ix in -1:1
                    tmp_dist = 0.0
                    tmp_dist += (param.L*ix + r1[1] - r2[1])^2
                    tmp_dist += (param.L*iy + r1[2] - r2[2])^2
                    tmp_dist += (param.L*iz + r1[3] - r2[3])^2
                    tmp_dist = sqrt(tmp_dist)
                    if tmp_dist < dist_old
                        dist_old = tmp_dist
                    end
                end
            end
        end
        return dist_old
    end

# Jastrow factor
    function func_u(param::Param, r::Float64)
        uofr = 0.0
        if r > param.rcut
            uofr = (param.a1/r)^param.a2
        else
            uofr = (param.a1/param.rcut)^param.a2
        end
        return uofr
    end

# radial two-point distribution function
    function accum_Nrdr(param::Param, Ndr, Nrdr)
        rmax = 0.5*param.L
        dr = rmax/Ndr
        for m in 1:Ndr
            r = rmax*(m-0.5)/Ndr
            for j in 1:(param.N-1)
                for i in (j+1):param.N
                    if (param.dist[i,j] >= r - 0.5*dr) && (param.dist[i,j] < r + 0.5*dr)
                        Nrdr[m] += 1.0
                    end
                end
            end
        end
    end

    function calc_gofr(param::Param, Nsample, Ndr, Nrdr, gofr, vecr)
        Ω = param.L^3
        ρ = param.N*1.0 / Ω
        rmax = 0.5*param.L
        dr = rmax/Ndr
        for m in 1:Ndr
            r = rmax*(m-0.5)/Ndr
            vecr[m] = r
            gofr[m] = Nrdr[m]/(2.0*Nsample*π*(ρ^2)*Ω*(r^2)*dr)
        end
    end

end