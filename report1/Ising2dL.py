import numpy as np
from Ising_lib import *
from matplotlib import pyplot

import argparse


def parse_args():
    Tc = 2.0/np.log(1.0+np.sqrt(2.0)) 
    parser = argparse.ArgumentParser(description='Monte Carlo simulation of the square lattice Ising model')
    parser.add_argument('-L',metavar='L',dest='L', type=int, default=16,
                        help='the size of square lattice. (default: L = 16)')
    parser.add_argument('-t', '--thermalization',metavar='thermalization',dest='thermalization', type=int,default=10000,
                        help='MC steps for thermalization. (default: 10000)')
    parser.add_argument('-o', '--observation',metavar='observation',dest='observation', type= int,default=50000,
                        help='MC steps for observation. (default: 50000)')
    parser.add_argument('-T', '--Temperature',metavar='T',dest='T', type=float,default=Tc,
                        help='Temperature. (default: T= Tc)')
    parser.add_argument('-hz', metavar='hz',dest='hz', type=float,default=0.0,
                        help='External magnetic field. (default: h= 0)')
    parser.add_argument( '-a','--algorithm', metavar='algorithm',dest='algorithm',default="metropolis",
                         help='Algorithms for MC simulation. You can use "metropolis", "heatbath" or "cluster"(Swendsen-Wang) (default: metropolis)')
    parser.add_argument('-s', '--seed',metavar='seed',dest='seed', type=int,default=11,
                        help='seed for random number generator. (default: seed= 11)')
    parser.add_argument('-c', '--correlation_time',metavar='correlation_time',dest='correlation_time', type=int,default=700,
                        help=' The longest time (MC steps) for plotting the auto-correlation fuctions. (default: correlation_time=500)')

    return parser.parse_args()

#Parameters for calculation
Tc = 2.0/np.log(1.0+np.sqrt(2.0)) ## The critical temperature of the Ising model

args = parse_args()
L1, L2, L3 = [16, 64, 256]
N_1 = L1**2
N_2 = L2**2
N_3 = L3**2
T = args.T
h = args.hz

algorithm = args.algorithm
random_seed = args.seed
thermalization = args.thermalization
observation = args.observation

correlation_time = args.correlation_time

print("## Simulation conditions:")
if algorithm == "heatbath":
    print("## Algorithm = Heat bath")
elif algorithm == "cluster":
    print("## Algorithm = Swendsen-Wang")
else:
    print("## Algorithm = Metropolis")
print("## L1 = "+repr(L1))
print("## L2 = "+repr(L2))
print("## L3 = "+repr(L3))
print("## T = "+repr(T))
print("## h = "+repr(h))
print("## random seed = "+repr(random_seed))
print("## thermalization steps = "+repr(thermalization))
print("## observation steps = "+repr(observation))
print("## correlation_time = "+repr(correlation_time))

## Initialization and run simulation
mag_1, mag2_1, mag2_imp_1, mag4_1, mag4_imp_1, mag_abs_1, ene_1, ene2_1 = MC(L1,T,h,thermalization,observation,random_seed,algorithm)
mag_2, mag2_2, mag2_imp_2, mag4_2, mag4_imp_2, mag_abs_2, ene_2, ene2_2 = MC(L2,T,h,thermalization,observation,random_seed,algorithm)
mag_3, mag2_3, mag2_imp_3, mag4_3, mag4_imp_3, mag_abs_3, ene_3, ene2_3 = MC(L3,T,h,thermalization,observation,random_seed,algorithm)

## Output with error estimated by Jackknife method
def variance(e,e2):
    return e2 -e**2
def binder(m2,m4):
    return m4 / m2 **2

E_1, E_err_1 = Jackknife(ene_1,bin_size=max(100,observation//100))
E2_1,E2_err_1 = Jackknife(ene2_1,bin_size=max(100,observation//100))
M_1,M_err_1 = Jackknife(mag_1,bin_size=max(100,observation//100))
M2_1,M2_err = Jackknife(mag2_1,bin_size=max(100,observation//100))
M4_1,M4_err = Jackknife(mag4_1,bin_size=max(100,observation//100))
C_1, C_err_1 = Jackknife(ene_1,bin_size=max(100,observation//100),func=variance, data2=ene2_1)
C_1 *= N_1/T**2
C_err_1 *= N_1/T**2
b_1, b_err_1 = Jackknife(mag2_1,bin_size=max(100,observation//100),func=binder, data2=mag4_1)

E_2, E_err_2 = Jackknife(ene_2,bin_size=max(100,observation//100))
E2_2,E2_err_2 = Jackknife(ene2_2,bin_size=max(100,observation//100))
M_2,M_err_2 = Jackknife(mag_2,bin_size=max(100,observation//100))
M2_2,M2_err = Jackknife(mag2_2,bin_size=max(100,observation//100))
M4_2,M4_err = Jackknife(mag4_2,bin_size=max(100,observation//100))
C_2, C_err_2 = Jackknife(ene_2,bin_size=max(100,observation//100),func=variance, data2=ene2_2)
C_2 *= N_2/T**2
C_err_2 *= N_2/T**2
b_2, b_err_2 = Jackknife(mag2_2,bin_size=max(100,observation//100),func=binder, data2=mag4_2)

E_3, E_err_3 = Jackknife(ene_3,bin_size=max(100,observation//100))
E2_3,E2_err_3 = Jackknife(ene2_3,bin_size=max(100,observation//100))
M_3,M_err_3 = Jackknife(mag_3,bin_size=max(100,observation//100))
M2_3,M2_err = Jackknife(mag2_3,bin_size=max(100,observation//100))
M4_3,M4_err = Jackknife(mag4_3,bin_size=max(100,observation//100))
C_3, C_err_3 = Jackknife(ene_3,bin_size=max(100,observation//100),func=variance, data2=ene2_3)
C_3 *= N_3/T**2
C_err_3 *= N_3/T**2
b_3, b_err_3 = Jackknife(mag2_3,bin_size=max(100,observation//100),func=binder, data2=mag4_3)

print ("[Outputs with errors estimated by Jackknife method]")
print ("L = "+repr(L1))
print ("T = " + repr(T))
print ("Energy = " + repr(E_1) + " +- " +repr(E_err_1))
print ("Energy^2 = " + repr(E2_1) + " +- " +repr(E2_err_1))
print ("Magnetization = " + repr(M_1) + " +- " +repr(M_err_1))
print ("Magnetization^2 = " + repr(M2_1) + " +- " +repr(M2_err))
print ("Magnetization^4 = " + repr(M4_1) + " +- " +repr(M4_err))
print ("Specific heat = " + repr(C_1) + " +- " +repr(C_err_1))
print ("Susceptibility = " + repr(M2_1/T * N_1) + " +- " +repr(M2_err/T * N_1))
print ("Binder ratio = " + repr(b_1) + " +- " +repr(b_err_1))

print ("[Outputs with errors estimated by Jackknife method]")
print ("L = "+repr(L2))
print ("T = " + repr(T))
print ("Energy = " + repr(E_2) + " +- " +repr(E_err_2))
print ("Energy^2 = " + repr(E2_2) + " +- " +repr(E2_err_2))
print ("Magnetization = " + repr(M_2) + " +- " +repr(M_err_2))
print ("Magnetization^2 = " + repr(M2_2) + " +- " +repr(M2_err))
print ("Magnetization^4 = " + repr(M4_2) + " +- " +repr(M4_err))
print ("Specific heat = " + repr(C_2) + " +- " +repr(C_err_2))
print ("Susceptibility = " + repr(M2_2/T * N_2) + " +- " +repr(M2_err/T * N_2))
print ("Binder ratio = " + repr(b_2) + " +- " +repr(b_err_2))

print ("[Outputs with errors estimated by Jackknife method]")
print ("L = "+repr(L3))
print ("T = " + repr(T))
print ("Energy = " + repr(E_3) + " +- " +repr(E_err_3))
print ("Energy^2 = " + repr(E2_3) + " +- " +repr(E2_err_3))
print ("Magnetization = " + repr(M_3) + " +- " +repr(M_err_3))
print ("Magnetization^2 = " + repr(M2_3) + " +- " +repr(M2_err))
print ("Magnetization^4 = " + repr(M4_3) + " +- " +repr(M4_err))
print ("Specific heat = " + repr(C_3) + " +- " +repr(C_err_3))
print ("Susceptibility = " + repr(M2_3/T * N_3) + " +- " +repr(M2_err/T * N_3))
print ("Binder ratio = " + repr(b_3) + " +- " +repr(b_err_3))

if algorithm == "cluster":
    print("=================== L {} =========================".format(L1))
    M2_imp_1, M2_imp_err_1 = Jackknife(mag2_imp_1,bin_size=max(100,observation//100))
    print ("Magnetization^2: improved estimator = " + repr(M2_imp_1) + " +- " +repr(M2_imp_err_1))
    print ("Susceptibility: improved estimator  = " + repr(M2_imp_1/T * N_1) + " +- " +repr(M2_imp_err_1/T * N_1))
    b_imp_1, b_imp_err_1 = Jackknife(mag2_imp_1,bin_size=max(100,observation//100),func=binder, data2=mag4_imp_1)
    print ("Bindar ratio: improved estimator = " + repr(b_imp_1) + " +- " +repr(b_imp_err_1))

    print("=================== L {} =========================".format(L2))
    M2_imp_2, M2_imp_err_2 = Jackknife(mag2_imp_2,bin_size=max(100,observation//100))
    print ("Magnetization^2: improved estimator = " + repr(M2_imp_2) + " +- " +repr(M2_imp_err_2))
    print ("Susceptibility: improved estimator  = " + repr(M2_imp_2/T * N_2) + " +- " +repr(M2_imp_err_2/T * N_2))
    b_imp_2, b_imp_err_2 = Jackknife(mag2_imp_2,bin_size=max(100,observation//100),func=binder, data2=mag4_imp_2)
    print ("Bindar ratio: improved estimator = " + repr(b_imp_2) + " +- " +repr(b_imp_err_2))

    print("=================== L {} =========================".format(L3))
    M2_imp_3, M2_imp_err_3 = Jackknife(mag2_imp_3,bin_size=max(100,observation//100))
    print ("Magnetization^2: improved estimator = " + repr(M2_imp_3) + " +- " +repr(M2_imp_err_3))
    print ("Susceptibility: improved estimator  = " + repr(M2_imp_3/T * N_3) + " +- " +repr(M2_imp_err_3/T * N_3))
    b_imp_3, b_imp_err_3 = Jackknife(mag2_imp_3,bin_size=max(100,observation//100),func=binder, data2=mag4_imp_3)
    print ("Bindar ratio: improved estimator = " + repr(b_imp_3) + " +- " +repr(b_imp_err_3))


pyplot.figure()
pyplot.title("L= " + repr(L1)+" Ising model:"+ " time series")
pyplot.xlabel("MC steps")
pyplot.ylabel("Energy")
pyplot.plot(np.arange(ene_1.size),ene_1,".")
pyplot.savefig('ene_L1.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L2)+" Ising model:"+ " time series")
pyplot.xlabel("MC steps")
pyplot.ylabel("Energy")
pyplot.plot(np.arange(ene_2.size),ene_2,".")
pyplot.savefig('ene_L2.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L3)+" Ising model:"+ " time series")
pyplot.xlabel("MC steps")
pyplot.ylabel("Energy")
pyplot.plot(np.arange(ene_3.size),ene_3,".")
pyplot.savefig('ene_L3.jpg')
pyplot.clf()

mag_d_1 = mag_1 - mag_1.mean()
mag2_d_1 = mag2_1 - mag2_1.mean()
ene_d_1 = ene_1 - ene_1.mean()

mag_d_2 = mag_2 - mag_2.mean()
mag2_d_2 = mag2_2 - mag2_2.mean()
ene_d_2 = ene_2 - ene_2.mean()

mag_d_3 = mag_3 - mag_3.mean()
mag2_d_3 = mag2_3 - mag2_3.mean()
ene_d_3 = ene_3 - ene_3.mean()

cor_mag_1 = np.correlate(mag_d_1,mag_d_1,mode="full")
cor_mag_1 = cor_mag_1[cor_mag_1.size//2:]/np.arange(mag_d_1.size,0,-1)
cor_mag2_1 = np.correlate(mag2_d_1,mag2_d_1,mode="full")
cor_mag2_1 = cor_mag2_1[cor_mag2_1.size//2:]/np.arange(mag2_d_1.size,0,-1)
cor_ene_1 = np.correlate(ene_d_1,ene_d_1,mode="full")
cor_ene_1 = cor_ene_1[cor_ene_1.size//2:]/np.arange(ene_d_1.size,0,-1)

cor_mag_2 = np.correlate(mag_d_2,mag_d_2,mode="full")
cor_mag_2 = cor_mag_2[cor_mag_2.size//2:]/np.arange(mag_d_2.size,0,-1)
cor_mag2_2 = np.correlate(mag2_d_2,mag2_d_2,mode="full")
cor_mag2_2 = cor_mag2_2[cor_mag2_2.size//2:]/np.arange(mag2_d_2.size,0,-1)
cor_ene_2 = np.correlate(ene_d_2,ene_d_2,mode="full")
cor_ene_2 = cor_ene_2[cor_ene_2.size//2:]/np.arange(ene_d_2.size,0,-1)

cor_mag_3 = np.correlate(mag_d_3,mag_d_3,mode="full")
cor_mag_3 = cor_mag_3[cor_mag_3.size//2:]/np.arange(mag_d_3.size,0,-1)
cor_mag2_3 = np.correlate(mag2_d_3,mag2_d_3,mode="full")
cor_mag2_3 = cor_mag2_3[cor_mag2_3.size//2:]/np.arange(mag2_d_3.size,0,-1)
cor_ene_3 = np.correlate(ene_d_3,ene_d_3,mode="full")
cor_ene_3 = cor_ene_3[cor_ene_3.size//2:]/np.arange(ene_d_3.size,0,-1)

pyplot.figure()
pyplot.title("L= " + repr(L1)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_M(t)$")
pyplot.plot(np.arange(cor_mag_1.size),cor_mag_1)
pyplot.xlim([0,correlation_time])
# pyplot.xlim([0,100])
pyplot.subplots_adjust(left=0.2)
pyplot.savefig('C_M(t)_L1.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L1)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_M2(t)$")
pyplot.plot(np.arange(cor_mag2_1.size),cor_mag2_1)
pyplot.xlim([0,correlation_time])
# pyplot.xlim([0,100])
pyplot.subplots_adjust(left=0.2)
pyplot.savefig('C_M2(t)_L1.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L2)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_M(t)$")
pyplot.plot(np.arange(cor_mag_2.size),cor_mag_2)
pyplot.xlim([0,correlation_time])
# pyplot.xlim([0,100])
pyplot.subplots_adjust(left=0.2)
pyplot.savefig('C_M(t)_L2.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L2)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_M2(t)$")
pyplot.plot(np.arange(cor_mag2_2.size),cor_mag2_2)
pyplot.xlim([0,correlation_time])
# pyplot.xlim([0,100])
pyplot.subplots_adjust(left=0.2)
pyplot.savefig('C_M2(t)_L2.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L3)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_M(t)$")
pyplot.plot(np.arange(cor_mag_3.size),cor_mag_3)
pyplot.xlim([0,correlation_time])
# pyplot.xlim([0,100])
pyplot.subplots_adjust(left=0.2)
pyplot.savefig('C_M(t)_L3.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L3)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_M2(t)$")
pyplot.plot(np.arange(cor_mag2_3.size),cor_mag2_3)
pyplot.xlim([0,correlation_time])
# pyplot.xlim([0,100])
pyplot.subplots_adjust(left=0.2)
pyplot.savefig('C_M2(t)_L3.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L1)+","+repr(L2)+","+repr(L3)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_M2(t)$")
pyplot.plot(np.arange(cor_mag2_1.size),cor_mag2_1, label='{}'.format("$L={}$".format(L1)), color='red')
pyplot.plot(np.arange(cor_mag2_2.size),cor_mag2_2, label='{}'.format("$L={}$".format(L2)), color='blue')
pyplot.plot(np.arange(cor_mag2_3.size),cor_mag2_3, label='{}'.format("$L={}$".format(L3)), color='yellow')
pyplot.legend()
pyplot.xlim([0,correlation_time])
# pyplot.xlim([0,100])
pyplot.subplots_adjust(left=0.2)
pyplot.savefig('C_M2(t)_L1L2L3.jpg')
pyplot.clf()


if algorithm == "cluster":
    mag2_imp_d_1 = mag2_imp_1 - mag2_imp_1.mean()
    cor_mag2_imp_1 = np.correlate(mag2_imp_d_1,mag2_imp_d_1,mode="full")
    cor_mag2_imp_1 = cor_mag2_imp_1[cor_mag2_imp_1.size//2:]/np.arange(mag2_imp_d_1.size,0,-1)

    pyplot.figure()
    pyplot.title("L= " + repr(L1)+" Ising model:"+ " autocorrelation")
    pyplot.xlabel("$Time$")
    pyplot.ylabel("$C_M2_imp(t)$")
    pyplot.plot(np.arange(cor_mag2_imp_1.size),cor_mag2_imp_1)
    pyplot.xlim([0,correlation_time])
    pyplot.savefig('C_M2_imp(t)_L1.jpg')
    pyplot.clf()

    mag2_imp_d_2 = mag2_imp_2 - mag2_imp_2.mean()
    cor_mag2_imp_2 = np.correlate(mag2_imp_d_2,mag2_imp_d_2,mode="full")
    cor_mag2_imp_2 = cor_mag2_imp_2[cor_mag2_imp_2.size//2:]/np.arange(mag2_imp_d_2.size,0,-1)

    pyplot.figure()
    pyplot.title("L= " + repr(L2)+" Ising model:"+ " autocorrelation")
    pyplot.xlabel("$Time$")
    pyplot.ylabel("$C_M2_imp(t)$")
    pyplot.plot(np.arange(cor_mag2_imp_2.size),cor_mag2_imp_2)
    pyplot.xlim([0,correlation_time])
    pyplot.savefig('C_M2_imp(t)_L2.jpg')
    pyplot.clf()

    mag2_imp_d_3 = mag2_imp_3 - mag2_imp_3.mean()
    cor_mag2_imp_3 = np.correlate(mag2_imp_d_3,mag2_imp_d_3,mode="full")
    cor_mag2_imp_3 = cor_mag2_imp_3[cor_mag2_imp_3.size//2:]/np.arange(mag2_imp_d_3.size,0,-1)

    pyplot.figure()
    pyplot.title("L= " + repr(L3)+" Ising model:"+ " autocorrelation")
    pyplot.xlabel("$Time$")
    pyplot.ylabel("$C_M2_imp(t)$")
    pyplot.plot(np.arange(cor_mag2_imp_3.size),cor_mag2_imp_3)
    pyplot.xlim([0,correlation_time])
    pyplot.savefig('C_M2_imp(t)_L3.jpg')
    pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L1)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_E(t)$")
pyplot.plot(np.arange(cor_ene_1.size),cor_ene_1)
pyplot.xlim([0,correlation_time])
pyplot.subplots_adjust(left=0.2)
pyplot.savefig('C_E(t)_L1.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L2)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_E(t)$")
pyplot.plot(np.arange(cor_ene_2.size),cor_ene_2)
pyplot.xlim([0,correlation_time])
pyplot.savefig('C_E(t)_L2.jpg')
pyplot.clf()

pyplot.figure()
pyplot.title("L= " + repr(L3)+" Ising model:"+ " autocorrelation")
pyplot.xlabel("$Time$")
pyplot.ylabel("$C_E(t)$")
pyplot.plot(np.arange(cor_ene_3.size),cor_ene_3)
pyplot.xlim([0,correlation_time])
pyplot.savefig('C_E(t)_L3.jpg')
pyplot.clf()

