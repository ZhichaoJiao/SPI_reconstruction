/*
Writen by Zhi Geng and Zhichao Jiao
*/


#include <hdf5.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdarg.h>
#include <getopt.h>
#include "nfft3.h"
#define NFFT_PRECISION_double

int main(int argc, char *argv[])
{ // 输入电子密度数据density.txt,光源和探测器相关参数,输出经过傅立叶变换后的三维Volume,且各个衍射点强度是振幅的平方.
    /*声明各个参数变量，并赋予初始值*/
    char file_in[400];
    char file_out[400];
    double z_det = 1;
    double lambda = 1.0e-10;
    double pix_len = 75.0e-6;
    int na = 0, nb = 0, nc = 0; // 3D dimendsions of the input density
    int size = 512;                   // Size of Valume
    double DETA = 1e-10;              // density.txt的采样间隔

    double dsize = size;
    double center = (dsize - 1) / 2;
    double x_plane; // 正空间探测器上(center,0)像素点的x坐标,单位是m.
    double q_max;   // 倒空间Volume中最边缘一个像素点(0,center,center)的x坐标,单位是1/m.
    float(*volume_pow)[size][size] = (float(*)[size][size])malloc(sizeof(float) * size * size * size);
    memset(volume_pow, 0, sizeof(float) * size * size * size);
    float(*volume_sqrt)[size][size] = (float(*)[size][size])malloc(sizeof(float) * size * size * size);
    memset(volume_sqrt, 0, sizeof(float) * size * size * size);
    float(*volume_phase)[size][size] = (float(*)[size][size])malloc(sizeof(float) * size * size * size);
    memset(volume_phase, 0, sizeof(float) * size * size * size);

    /*used for nfft*/
    fftw_complex *data;
    nfft_plan p;
    int d = 3;
    int N[3], n[3];
    int M, m;

    int i, j, k, index;

    /*Long options*/
    const struct option longopts[] = {
        {"density_file", 1, NULL, 'i'},
        {"file_out", 1, NULL, 'o'},
        {"z_det", 1, NULL, 2},
        {"lambda", 1, NULL, 3},
        {"pix_len", 1, NULL, 4},
        {"size", 1, NULL, 5},
        {"na", 1, NULL, 14},
        {"nb", 1, NULL, 15},
        {"nc", 1, NULL, 16},

        {0, 0, NULL, 0}};

    /*short options*/
    /*从命令中读取参数信息，如果有输入，那么将输入赋给参数，如果没有输入，那么使用参数变量声明时候的初始值*/
    int c;
    char *rval;
    while ((c = getopt_long(argc, argv, "io:", longopts, NULL)) != -1)
    {
        switch (c)
        {
        case 'i':
            snprintf(file_in, 400, "%s", optarg);
            break;

        case 'o':
            snprintf(file_out, 400, "%s", optarg);
            break;

        case 2:
            z_det = strtod(optarg, &rval);
            break;

        case 3:
            lambda = strtod(optarg, &rval);
            break;

        case 4:
            pix_len = strtod(optarg, &rval);
            break;

        case 5:
            size = strtol(optarg, &rval, 10);
            break;

        case 14:
            na = strtol(optarg, &rval, 10);
            break;

        case 15:
            nb = strtol(optarg, &rval, 10);
            break;

        case 16:
            nc = strtol(optarg, &rval, 10);
            break;
        }
    }

    /*read in 3D electron density*/
    data = (fftw_complex *)nfft_malloc(sizeof(fftw_complex) * na * nb * nc);

    FILE *fp = fopen(file_in, "r");

    for (k = 0; k < nc; k++)
        for (j = 0; j < nb; j++)
            for (i = 0; i < na; i++)
            {
                double real;
                fscanf(fp, "%le\n", &real);
                data[k * na * nb + j * na + i] = real;
            }
    /*prepare for nfft*/
    M = size * size * size;
    N[0] = na;
    N[1] = nb;
    N[2] = nc;
    n[0] = (int)nfft_next_power_of_2(N[0]) * 2; // 最靠近N[0]的2的幂指数
    n[1] = (int)nfft_next_power_of_2(N[1]) * 2;
    n[2] = (int)nfft_next_power_of_2(N[2]) * 2;

    printf("\n   NFFT initializing......\n");

    nfft_init_guru(&p, 3, N, M, n, 6,
                   PRE_PHI_HUT | PRE_PSI | MALLOC_X | MALLOC_F_HAT |
                       MALLOC_F | FFTW_INIT | FFT_OUT_OF_PLACE,
                   FFTW_MEASURE | FFTW_DESTROY_INPUT);

    /* since p.x is arrayed in sequence X->Y->Z, meaning X is first stored, followed by Y and Z, */
    /* p.f_hat must be arrayed inverted, meaning Z is first stored, followed ny Y and X */
    for (i = 0; i < na * nb * nc; i++)
    {
        int col = (i % (na * nb)) % na;                    // along X
        int row = (i % (na * nb)) / na;                    // along Y
        int slc = i / (na * nb);                           // along Z
        p.f_hat[col * nb * nc + row * nc + slc] = data[i]; // 正空间均匀采样的数据
    }

    // 通过探测器上面(center,0)点,计算q_max.
    x_plane = center * pix_len;
    q_max = fabs(x_plane / (sqrt(x_plane * x_plane + z_det * z_det) * lambda));

    /*loop for each pixel*/
    printf("\n   Imput p.x......\n");
    for (k = 0; k < size; k++)
    {
        for (j = 0; j < size; j++)
        {
            for (i = 0; i < size; i++)
            {
                p.x[3 * (k * size * size + j * size + i)] = (i - center) * q_max * DETA / center;
                p.x[3 * (k * size * size + j * size + i) + 1] = (j - center) * q_max * DETA / center;
                p.x[3 * (k * size * size + j * size + i) + 2] = (k - center) * q_max * DETA / center;
            }
        }
    }

    printf("\n   Precompute one psi......\n");
    if (p.flags & PRE_ONE_PSI)
    {
        nfft_precompute_one_psi(&p);
    }

    /*conduct NFFT*/
    printf("\n   Conduct NFFT......\n");
    nfft_trafo(&p);

    /*calculate volume*/
    printf("\n   Calculate volume......\n");
    for (k = 0; k < size; k++)
    {
        for (j = 0; j < size; j++)
        {
            for (i = 0; i < size; i++)
            {
                index = k * size * size + j * size + i;
                volume_pow[i][j][k] = pow(creal(p.f[index]), 2) + pow(cimag(p.f[index]), 2);
                volume_sqrt[i][j][k] = sqrt(pow(creal(p.f[index]), 2) + pow(cimag(p.f[index]), 2));
                volume_phase[i][j][k] = atan2(cimag(p.f[index]), creal(p.f[index]));
            }
        }
    }

    /*save into file*/
    printf("\n   Save volume......\n");
    hid_t file_id, dset_id, dspace_id;
    hsize_t dims[3] = {size, size, size};
    herr_t status;

    file_id = H5Fcreate(file_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    dspace_id = H5Screate_simple(3, dims, NULL);
    dset_id = H5Dcreate(file_id, "volume_pow", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, volume_pow);
    H5Dclose(dset_id);
    dset_id = H5Dcreate(file_id, "volume_sqrt", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, volume_sqrt);
    H5Dclose(dset_id);
    dset_id = H5Dcreate(file_id, "phase", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, volume_phase);
    H5Dclose(dset_id);
    H5Sclose(dspace_id);

    H5Fclose(file_id);

    nfft_finalize(&p);

    printf("\nFinish ! ! !\n");

    fclose(fp);
    nfft_free(data);
    free(volume_pow);
    free(volume_sqrt);
    free(volume_phase);

    return 0;
}
