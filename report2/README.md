# Report2

Before you run anything, ensure that there are ten directories named 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 in the current directory.

- The files 10_mc.jl, sample_vmc_helium4.jl, and Makefile are necessary for generating Figure 1. Before executing this code, make sure that you have altered several 'N' values and carried out ten simulations for each 'N'. 

- The file 10_mc_Nsample.jl is needed to generate Figure 2.

- The file N12800_N25600.jl is necessary for creating Figure 3.

- The file N25600_N51200.jl is required to generate Figure 4.

- The file N51200_N102400.jl is required to produce Figure 5.

For Problem 2, ensure that mkl.jl works before executing any code. 

- The file sample_Hubbard_small_v0.jl implements LANCZOS and LOBCG and the part that directly solves the eigenvalues with LAPACK (via mkl.jl). This code also generates Figure 6.

- The file a1.jl is needed to generate Figure 7.

- To generate Figures 8 and 9, you need demo.def, create_def.py, a2.py, and a2.jl.

This serves as a guide to successfully generate the required figures for the project. Please ensure that all dependencies are properly installed and working. Thank you for using this code repository!