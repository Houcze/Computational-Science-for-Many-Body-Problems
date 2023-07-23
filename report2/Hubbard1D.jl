module Hubbard1D
#
  mutable struct Param
    t::Float64
    U::Float64
    L::Int64
    Param() = new()
  end
#
  function initialize(param::Param, t, U, L)
    param.t = t
    param.U = U
    param.L = L
  end
#
function parity(m)
    m ⊻= m>>1 # XOR
    m ⊻= m>>2 # XOR
    m = (m&Int64(0x1111111111111111) ) *  Int64(0x1111111111111111)
    return (m>>60)&1
end
#
function Cop(ell,i,c0)
  icomb = 2^ell
  if parity(i&icomb) == 1
    return i,0.0
  else
    j = i⊻icomb
    sgn = 1.0 - 2.0*parity(i&(icomb-1))
    return j, c0 * sgn
  end
end
#
function Aop(ell,i,c0)
  icomb = 2^ell
  if parity(i&icomb) == 0
    return i,0.0
  else
    j = i⊻icomb
    sgn = 1.0 - 2.0*parity(i&(icomb-1))
    return j, c0 * sgn
  end
end
# Hubbard hamiltonian
function multiply(param::Param,v0,v1)
    for k = 1:2^(2*param.L)
        c0 = v0[k]
        i = k - 1
        # define 1D Hubbard
        for ell = 0:param.L-1
          j1, c1 = Aop(2*ell,i,c0)
          j2, c2 = Cop(2*mod(ell+1,param.L),j1,c1)
          v1[j2+1] -= param.t * c2
          j1, c1 = Aop(2*ell,i,c0)
          j2, c2 = Cop(2*mod(ell-1,param.L),j1,c1)
          v1[j2+1] -= param.t * c2
          j1, c1 = Aop(2*ell+1,i,c0)
          j2, c2 = Cop(2*mod(ell+1,param.L)+1,j1,c1)
          v1[j2+1] -= param.t * c2
          j1, c1 = Aop(2*ell+1,i,c0)
          j2, c2 = Cop(2*mod(ell-1,param.L)+1,j1,c1)
          v1[j2+1] -= param.t * c2
          j1, c1 = Aop(2*ell,i,c0)
          j2, c2 = Cop(2*ell,j1,c1)
          j1, c1 = Aop(2*ell+1,j2,c2)
          j2, c2 = Cop(2*ell+1,j1,c1)
          v1[j2+1] += param.U * c2
        end
    end
    return 0
end
#
function countbit(i,length)
    j = 0
    for k = 0:length-1
      icomb = 2^k
      j += parity(i&icomb)
    end
    return j
end
#
end