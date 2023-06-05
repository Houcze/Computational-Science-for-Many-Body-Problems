/*
 coding: utf-8

  Exercise code for Monte Calro simulation of the 2d Ising model
 This module is for simulation of 2d Ising model on the square lattice, whose Hamiltonian is given by
 $$ \mathcal{H} = -J \sum_{\langle i,j\rangle} S_i S_j - h \sum_i S_i ,$$
 where $S_i = \pm 1$.

 You can select three simulation algorithms:
 * metropolis
 * heatbath
 * cluster (Swendsen-Wang)

 The outputs are:
 * Energy: $\langle E\rangle = \langle \mathcal{H}\rangle/N$.
 * Squared magnetization: $\langle M^2\rangle = \langle (\sum_i S_i)^2\rangle/N^2$.
 * Specific heat: $N(\langle E^2\rangle - \langle E\rangle^2)/T$
 * Magnetic susceptibility: $N(\langle M^2\rangle\rangle)/T$
 * Anothor Magnetic susceptibility: $N(\langle M^2\rangle - \langle |M|^2\rangle)/T$
 * Binder ratio: $(\langle M^4\rangle/\langle M^2\rangle)/T$
*/
#include <cmath>
#include <random>
#include <curand.h>
#include <iostream>
#include <algorithm>
#include <fstream>

static const int threadsperblock = 256;

__global__ void metropolis_k(int *S, double *exps, int1 L, double *ran)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < L.x && iy < L.x)
    {
        int N = L.x * L.x;
        int hm = (S[ix * L.x + iy] * (S[((ix + 1) % L.x) * L.x + iy] + S[ix * L.x + (iy + 1) % L.x] + S[((ix - 1 + L.x) % L.x) * L.x + iy] + S[ix * L.x + ((iy - 1 + L.x) % L.x)]) + 4) / 2;
        if (ran[ix * L.x + iy] < exps[hm * 2 + (S[ix * L.x + iy] + 1) / 2])
        {
            S[ix * L.x + iy] *= -1;
        }
    }
}

int metropolis(int *S, double *exps, int L, int seed)
{
    double *ran;
    cudaMalloc(&ran, L * L * sizeof(double));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniformDouble(gen, ran, L * L);
    curandDestroyGenerator(gen);
    metropolis_k<<<ceil(L * L / threadsperblock), threadsperblock>>>(S, exps, make_int1(L), ran);
    cudaFree(ran);
    return EXIT_SUCCESS;
}

void Initialize(int seed, int L, int *S)
{
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, 1);
    std::vector<int> S_(L * L);
    std::generate(S_.begin(), S_.end(), [&](){ return -2 * distribution(generator) + 1; });
    std::copy(S_.begin(), S_.end(), S);
}

double Calc_local_energy(int *S, int L)
{
    double local_ene = 0.0;
    for (int ix = 0; ix < L; ix++)
    {
        for (int iy = 0; iy < L; iy++)
        {
            local_ene += S[ix * L + iy] * (S[(ix + 1) % L * L + iy] + S[ix * L + (iy + 1) % L]);
        }
    }
    return local_ene;
}

template <class T>
T sum(T *arr, int N)
{
    T result{0};
    for (int i = 0; i < N; i++)
    {
        result += arr[i];
    }
    return result;
}

void MC(int L, double T, double h, int thermalization, double *mag, double *mag2, double *mag4, double *mag_abs, double *ene, double *ene2, int observation, int seed = 11)
{
    int N = L * L;
    int *S;
    S = (int *)malloc(N * sizeof(int));
    Initialize(seed, L, S);

    double *exps;
    exps = (double *)malloc(5 * 2 * sizeof(double));

    double hm;
    double sh;
    for (int i = 0; i < 5; i++)
    {
        hm = -4.0 + 2.0 * i;
        for (int j = 0; j < 2; j++)
        {
            sh = h * (2 * j - 1);
            exps[i * 2 + j] = exp(-2.0 * (hm + sh) / T);
        }
    }

    double *exps_dev;
    int *S_dev;

    cudaMalloc(&exps_dev, 5 * 2 * sizeof(double));
    cudaMalloc(&S_dev, L * L * sizeof(int));

    cudaMemcpy(exps_dev, exps, 5 * 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(S_dev, S, L * L * sizeof(int), cudaMemcpyHostToDevice);
    int seed_;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis;
    for (int i = 0; i < thermalization; i++)
    {
        seed_ = dis(gen);
        metropolis(S_dev, exps_dev, L, seed_);
    }

    double local_mag;
    double local_ene;

    for (int i = 0; i < observation; i++)
    {
        seed_ = dis(gen);
        metropolis(S_dev, exps_dev, L, seed_);
        cudaMemcpy(S, S_dev, L * L * sizeof(int), cudaMemcpyDeviceToHost);
        local_mag = double(sum<int>(S, N)) / N;
        // std::cout << "local_mag is " << local_mag << std::endl;
        mag[i] = local_mag;
        mag2[i] = local_mag * local_mag;
        mag4[i] = local_mag * local_mag * local_mag * local_mag;
        mag_abs[i] = abs(local_mag);
        local_ene = -Calc_local_energy(S, L) / N - h * local_mag;
        // std::cout << "local_ene is " << local_ene << std::endl;
        ene[i] = local_ene;
        ene2[i] = local_ene * local_ene;
    }
    cudaFree(exps_dev);
    cudaFree(S_dev);
    free(exps);
}

void variance(double *e, double *e2, double *result, int N)
{
    for (int i = 0; i < N; i++)
    {
        result[i] = e2[i] - e[i] * e[i];
    }
}

void variance(int *e, int *e2, int *result, int N)
{
    for (int i = 0; i < N; i++)
    {
        result[i] = e2[i] - e[i] * e[i];
    }
}

void binder(double *m2, double *m4, double *result, int N)
{
    for (int i = 0; i < N; i++)
    {
        result[i] = m4[i] / (m2[i] * m2[i]);
    }
}

void make_bin(double *data, int bin_size_in, int data_size, double *bin_data, int *bin_size_out)
{
    *bin_size_out = bin_size_in;
    int bin_num = data_size / (*bin_size_out);
    if (bin_num < 10)
    {
        *bin_size_out = data_size / 10;
        bin_num = 10;
    }

    double *bin_data_temp = new double[bin_num];
    double *data_temp = new double[*bin_size_out];

    for (int i = 0; i < bin_num; i++)
    {
        std::copy(data + i * (*bin_size_out), data + (i + 1) * (*bin_size_out), data_temp);
        bin_data_temp[i] = sum<double>(data_temp, (*bin_size_out)) / (*bin_size_out);
    }

    double total = sum<double>(bin_data_temp, bin_num);

    for (int i = 0; i < bin_num; i++)
    {
        bin_data[i] = (total - bin_data_temp[i]) / (bin_num - 1);
    }

    delete[] bin_data_temp;
    delete[] data_temp;
}

void make_bin(int *data, int bin_size_in, int data_size, int *bin_data, int *bin_size_out)
{
    *bin_size_out = bin_size_in;
    int bin_num = data_size / (*bin_size_out);
    if (bin_num < 10)
    {
        *bin_size_out = data_size / 10;
        bin_num = 10;
    }

    int *bin_data_temp = new int[bin_num];
    int *data_temp = new int[*bin_size_out];

    for (int i = 0; i < bin_num; i++)
    {
        std::copy(data + i * (*bin_size_out), data + (i + 1) * (*bin_size_out), data_temp);
        bin_data_temp[i] = sum<int>(data_temp, (*bin_size_out)) / (*bin_size_out);
    }

    double total = sum<int>(bin_data_temp, bin_num);

    for (int i = 0; i < bin_num; i++)
    {
        bin_data[i] = (total - bin_data_temp[i]) / (bin_num - 1);
    }

    delete[] bin_data_temp;
    delete[] data_temp;
}

void square(double *arr1, double *arr2, int N)
{
    for (int i = 0; i < N; i++)
    {
        arr2[i] = arr1[i] * arr1[i];
    }
}

void square(int *arr1, int *arr2, int N)
{
    for (int i = 0; i < N; i++)
    {
        arr2[i] = arr1[i] * arr1[i];
    }
}

template <typename T>
void Jackknife(T *data, int bin_size, void (*func)(T *, T *, T *, int), T *data2, int data_size, double *average, double *error)
{
    int bin_size_out;
    T *bin_data = (T *)malloc((data_size / bin_size) * sizeof(T));
    T *bin_data2 = (T *)malloc((data_size / bin_size) * sizeof(T));
    T *f_result = (T *)malloc((data_size / bin_size) * sizeof(T));
    T *f_result2 = (T *)malloc((data_size / bin_size) * sizeof(T));

    make_bin(data, bin_size, data_size, bin_data, &bin_size_out);
    make_bin(data2, bin_size, data_size, bin_data2, &bin_size_out);
    func(bin_data, bin_data2, f_result, (data_size / bin_size));
    *average = sum<T>(f_result, (data_size / bin_size)) / (data_size / bin_size);
    square(f_result, f_result2, (data_size / bin_size));
    *error = sqrt((sum<T>(f_result2, (data_size / bin_size)) / (data_size / bin_size) - (*average) * (*average)) * ((data_size / bin_size) - 1));

    free(bin_data);
    free(bin_data2);
    free(f_result);
    free(f_result2);
}

template <typename T>
void Jackknife(T *data, int bin_size, int data_size, double *average, double *error)
{
    int bin_size_out;
    T *bin_data = (T *)malloc((data_size / bin_size) * sizeof(T));
    T *bin_data2 = (T *)malloc((data_size / bin_size) * sizeof(T));

    make_bin(data, bin_size, data_size, bin_data, &bin_size_out);

    *average = sum<T>(bin_data, (data_size / bin_size)) / (data_size / bin_size);
    square(bin_data, bin_data2, (data_size / bin_size));
    *error = sqrt((sum<T>(bin_data2, (data_size / bin_size)) / (data_size / bin_size) - (*average) * (*average)) * ((data_size / bin_size) - 1));

    free(bin_data);
    free(bin_data2);
}

void save_txt(double *array, int N, std::string name)
{
    std::ofstream file(name + ".txt");
    if (file.is_open())
    {
        for (int i = 0; i < N; ++i)
        {
            file << array[i] << '\n';
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }
}

int main(int argc, char *argv[])
{
    double *mag;
    double *mag2;
    double *mag4;
    double *mag_abs;
    double *ene;
    double *ene2;
    double Tc = 2.0 / log(1.0 + sqrt(2.0));

    int L = 16;
    int observation = 50000;
    int thermalization = 10000;
    double T = Tc;
    double h = 0.0;
    int seed = 111;

    int N = L * L;
    mag = (double *)malloc(observation * sizeof(double));
    mag2 = (double *)malloc(observation * sizeof(double));
    mag4 = (double *)malloc(observation * sizeof(double));
    mag_abs = (double *)malloc(observation * sizeof(double));
    ene = (double *)malloc(observation * sizeof(double));
    ene2 = (double *)malloc(observation * sizeof(double));

    std::cout << "## Algorithm = Metropolis\n";
    std::cout << "## L = " << L << '\n';
    std::cout << "## T = " << T << '\n';
    std::cout << "## h = " << h << '\n';
    std::cout << "## random seed = " << seed << '\n';
    std::cout << "## thermalization steps = " << thermalization << '\n';
    std::cout << "## observation steps = " << observation << '\n';

    MC(L, T, h, thermalization, mag, mag2, mag4, mag_abs, ene, ene2, observation, seed);

    save_txt(mag, observation, "mag");
    save_txt(mag2, observation, "mag2");
    save_txt(mag4, observation, "mag4");
    save_txt(mag_abs, observation, "mag_abs");
    save_txt(ene, observation, "ene");
    save_txt(ene2, observation, "ene2");

    double E, E_err;
    Jackknife<double>(ene, max(100, observation / 100), observation, &E, &E_err);
    double E2, E2_err;
    Jackknife<double>(ene2, max(100, observation / 100), observation, &E2, &E2_err);
    double M, M_err;
    Jackknife<double>(mag, max(100, observation / 100), observation, &M, &M_err);
    double M2, M2_err;
    Jackknife<double>(mag2, max(100, observation / 100), observation, &M2, &M2_err);
    double M4, M4_err;
    Jackknife<double>(mag4, max(100, observation / 100), observation, &M4, &M4_err);
    double C, C_err;
    Jackknife<double>(ene, max(100, observation / 100), variance, ene2, observation, &C, &C_err);
    C *= N / (T * T);
    C_err *= N / (T * T);
    double b, b_err;
    Jackknife<double>(mag2, max(100, observation / 100), binder, mag4, observation, &b, &b_err);

    std::cout << "### Outputs with errors estimated by Jackknife method ###" << '\n';
    std::cout << "T = " << T << '\n';
    std::cout << "Energy = " << E << " +- " << E_err << '\n';
    std::cout << "Energy^2 = " << E2 << " +- " << E2_err << '\n';
    std::cout << "Magnetization = " << M << " +- " << M_err << '\n';
    std::cout << "Magnetization^2 = " << M2 << " +- " << M2_err << '\n';
    std::cout << "Magnetization^4 = " << M4 << " +- " << M4_err << '\n';
    std::cout << "Specific heat = " << C << " +- " << C_err << '\n';
    std::cout << "Susceptibility = " << M2 / T * N << " +- " << M2_err / T * N << '\n';
    std::cout << "Binder ratio = " << b << " +- " << b_err << '\n';
}
