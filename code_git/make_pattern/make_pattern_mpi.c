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
#include <gsl/gsl_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#define NFFT_PRECISION_DOUBLE
#include "nfft3mp.h"
#include <omp.h>
#include <sched.h>
#include <mpi.h>

#define pi 3.141592653590

void GenPixels(double *pixels, double lambda, double z_det, double pix_len, int size);
void MakeRotMatrixEuler(double *rot_angle, double rot_matrix[3][3]);
int min(int x, int y);

int main(int argc, char *argv[])
{
    /*初始化mpi*/
    MPI_Init(NULL, NULL);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /*声明各个参数变量*/
    char file_in[400];
    char output_path[400];

    int n_pattern = 0;
    double z_det;        // sample to detector z_det in unit of meter
    double lambda;       // wavelength in unit of meter
    double pix_len;      // pixel size in unit of meter
    int na, nb, nc;      // 3D dimensions of the input density
    int size = 512;      // Size of Valume
    int beam_mask;       // radius of the beamstop in pixels
    int noise;           // whether add poisson noise
    double photon_num;   // 单位时间光子数
    double beamsize;     // 光斑直径,单位m,用于计算光通量
    double DETA = 1e-10; // density.txt的采样间隔

    /*Long options*/
    const struct option longopts[] = {
        {"density_file", 1, NULL, 'i'},
        {"output_path", 1, NULL, 'o'},
        {"z_det", 1, NULL, 2},
        {"lambda", 1, NULL, 3},
        {"pix_len", 1, NULL, 4},
        {"size", 1, NULL, 5},
        {"beam_mask", 1, NULL, 6},
        {"noise", 1, NULL, 7},
        {"n_pattern", 1, NULL, 11},
        {"photon_num", 1, NULL, 12},
        {"beamsize", 1, NULL, 13},
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
            snprintf(output_path, 400, "%s", optarg);
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

        case 6:
            beam_mask = strtol(optarg, &rval, 10);
            break;

        case 7:
            noise = strtol(optarg, &rval, 10);
            break;

        case 11:
            n_pattern = strtol(optarg, &rval, 10);
            break;

        case 12:
            photon_num = strtod(optarg, &rval);
            break;

        case 13:
            beamsize = strtod(optarg, &rval);
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

    // 输出参数
    if (0 == world_rank)
    {
        printf("\n");
        printf("file_in : %s\n", file_in);
        printf("output_path : %s\n", output_path);
        printf("n_pattern = %d\n", n_pattern);
        printf("lambda = %e\n", lambda);
        printf("photon_num = %e\n", photon_num);
        printf("beam_size = %e\n", beamsize);
        printf("size = %d\n", size);
        printf("z_det = %f\n", z_det);
        printf("pix_len = %e\n", pix_len);
        printf("beam_mask = %d\n", beam_mask);
        printf("na = %d    nb = %d     nc=%d\n", na, nb, nc);
        printf("noise = %d\n", noise);

        char parameter_file_path[400];
        FILE *parameter_file;
        snprintf(parameter_file_path, 400, "%s/parameter.txt", output_path);
        parameter_file = fopen(parameter_file_path, "w");
        fprintf(parameter_file, "file_in : %s\n", file_in);
        fprintf(parameter_file, "output_path : %s\n", output_path);
        fprintf(parameter_file, "N_pattern = %d\n", n_pattern);
        fprintf(parameter_file, "lambda = %e\n", lambda);
        fprintf(parameter_file, "photon_num = %e\n", photon_num);
        fprintf(parameter_file, "beam_size = %e\n", beamsize);
        fprintf(parameter_file, "size = %d\n", size);
        fprintf(parameter_file, "z_det = %f\n", z_det);
        fprintf(parameter_file, "pix_len = %e\n", pix_len);
        fprintf(parameter_file, "beam_mask = %d\n", beam_mask);
        fprintf(parameter_file, "na = %d   nb = %d   nc=%d\n", na, nb, nc);
        fprintf(parameter_file, "noise = %d\n", noise);
        fclose(parameter_file);
    }

    /*声明其他变量*/
    char file_out[400];
    double flux; // 光通量
    double dsize = size;
    double center = (dsize - 1) / 2;
    double x_plane;                                                      // 正空间探测器上(center,0)像素点的x坐标,单位是m.
    double q_max;                                                        // 倒空间Volume中最边缘一个像素点(0,center,center)的x坐标,单位是1/m.
    double alpha, beta, gamma;                                           // Eular angles
    double *pixels = (double *)malloc(sizeof(double) * size * size * 4); // 二维衍射图的坐标数据，已经变化到Ewald sphere上面，一维double数组，pixels = {x1, y1, z1, omega1, ..., }
    memset(pixels, 0, sizeof(double) * size * size * 4);
    float pattern[size][size];
    memset(pattern, 0, sizeof(float) * size * size);
    float phase[size][size];
    memset(phase, 0, sizeof(float) * size * size);
    double rot_angle[3] = {0}, rot_matrix[3][3] = {0}, rot_pix[3] = {0};
    double e_radius = 2.81794e-15;          // 经典电子半径,单位m
    double cubic;                           // 立体角
    int batch = n_pattern / world_size + 1; // 每个进程模拟的衍射图数量
    double(*pattern_angle_i)[4] = (double(*)[4])malloc(sizeof(double) * batch * 4);
    memset(pattern_angle_i, 0, sizeof(double) * batch * 4);
    double(*pattern_angle)[4] = (double(*)[4])malloc(sizeof(double) * (batch * world_size) * 4);
    memset(pattern_angle, 0, sizeof(double) * (batch * world_size) * 4);

    /*used for nfft*/
    fftw_complex *data;
    nfft_plan p;
    int d = 3;
    int N[3], n[3];
    int M, m;

    /*读取电子密度数据*/
    data = (fftw_complex *)nfft_malloc(sizeof(fftw_complex) * na * nb * nc);

    FILE *fp = fopen(file_in, "r");

    for (int k = 0; k < nc; k++)
        for (int j = 0; j < nb; j++)
            for (int i = 0; i < na; i++)
            {
                double real;
                fscanf(fp, "%le\n", &real);
                data[k * na * nb + j * na + i] = real;
            }

    /*初始化nfft参数*/
    M = size * size;
    N[0] = na;
    N[1] = nb;
    N[2] = nc;
    n[0] = (int)nfft_next_power_of_2(N[0]) * 2; // 最靠近N[0]的2的幂指数
    n[1] = (int)nfft_next_power_of_2(N[1]) * 2;
    n[2] = (int)nfft_next_power_of_2(N[2]) * 2;

    if (0 == world_rank)
        printf("\nNFFT initializing......\n");

    nfft_init_guru(&p, 3, N, M, n, 6,
                   PRE_PHI_HUT | PRE_PSI | MALLOC_X | MALLOC_F_HAT |
                       MALLOC_F | FFTW_INIT | FFT_OUT_OF_PLACE,
                   FFTW_MEASURE | FFTW_DESTROY_INPUT);

    /* since p.x is arrayed in sequence X->Y->Z, meaning X is first stored, followed by Y and Z, */
    /* p.f_hat must be arrayed inverted, meaning Z is first stored, followed ny Y and X */
    for (int i = 0; i < na * nb * nc; i++)
    {
        int col = (i % (na * nb)) % na;                    // along X
        int row = (i % (na * nb)) / na;                    // along Y
        int slc = i / (na * nb);                           // along Z
        p.f_hat[col * nb * nc + row * nc + slc] = data[i]; // 正空间均匀采样的数据
    }

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_rng_set(rng, 123456);

    GenPixels(pixels, lambda, z_det, pix_len, size);

    // 通过探测器上面(center,0)点,计算q_max.
    x_plane = center * pix_len;
    q_max = fabs(x_plane / (sqrt(x_plane * x_plane + z_det * z_det) * lambda));

    for (int nn = world_rank * batch; nn < min((world_rank + 1) * batch, n_pattern); nn++)
    {
        if (0 == world_rank)
        {
            printf("\rMake pattern_%d/%d......\n", nn, batch - 1);
            fflush(stdout);
        }
        /*generate random Eular angles*/
        rot_angle[0] = gsl_rng_uniform(rng) * 2 * pi;
        rot_angle[1] = gsl_rng_uniform(rng) * pi;
        rot_angle[2] = gsl_rng_uniform(rng) * 2 * pi;

        switch (nn)
        {
        case 0:
            rot_angle[0] = 0;
            rot_angle[1] = 0;
            rot_angle[2] = 0;
            break;

        case 1:
            rot_angle[0] = pi;
            rot_angle[1] = pi / 2;
            rot_angle[2] = 0;
            break;
        }

        int index = nn - world_rank * batch;
        pattern_angle_i[index][0] = nn;
        pattern_angle_i[index][1] = rot_angle[0];
        pattern_angle_i[index][2] = rot_angle[1];
        pattern_angle_i[index][3] = rot_angle[2];

        MakeRotMatrixEuler(rot_angle, rot_matrix);

        /*loop for each pixel*/
        for (int t = 0; t < size * size; t++)
        {
            for (int i = 0; i < 3; i++)
            {
                rot_pix[i] = 0;
                for (int j = 0; j < 3; j++)
                {
                    rot_pix[i] += rot_matrix[i][j] * pixels[t * 4 + j];
                }
            }
            int i = t / size;
            int j = t % size;
            p.x[3 * (size * j + i)] = rot_pix[0] * q_max * DETA / center;
            p.x[3 * (size * j + i) + 1] = rot_pix[1] * q_max * DETA / center;
            p.x[3 * (size * j + i) + 2] = rot_pix[2] * q_max * DETA / center;
        }

        if (p.flags & PRE_ONE_PSI)
        {
            nfft_precompute_one_psi(&p);
        }

        /*conduct NFFT*/
        nfft_trafo(&p);

        /*calculate real diffraction intensity*/
        for (int j = 0; j < size; j++)
        {
            for (int i = 0; i < size; i++)
            {

                if ((abs(j - center) < beam_mask) && (abs(i - center) < beam_mask))
                {
                    pattern[i][j] = 0;
                    continue;
                }
                int t = i * size + j;
                cubic = pixels[t * 4 + 3];
                flux = photon_num / (pi * beamsize * beamsize / 4);

                if (noise)
                {
                    pattern[i][j] = gsl_ran_poisson(rng, (pow(creal(p.f[j * size + i]), 2) + pow(cimag(p.f[j * size + i]), 2)) * cubic * flux * e_radius * e_radius); // p.f 是nfft后的衍射强度，是一个复数数组，元素与p.x一一对应。
                }
                else
                {
                    pattern[i][j] = (pow(creal(p.f[j * size + i]), 2) + pow(cimag(p.f[j * size + i]), 2)) * cubic * flux * e_radius * e_radius;
                }

                // 对前三张衍射图计算相位
                if (0 == nn || 1 == nn || 2 == nn)
                {
                    phase[i][j] = atan2(cimag(p.f[j * size + i]), creal(p.f[j * size + i]));
                }
            }
        }

        /*save into file*/

        snprintf(file_out, 400, "%s/pattern_%d.h5", output_path, nn);

        hid_t file_id, dset_id, dspace_id;
        hsize_t dims[2] = {size, size};
        hsize_t dims_angle[1] = {3};
        herr_t status;

        file_id = H5Fcreate(file_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        dspace_id = H5Screate_simple(2, dims, NULL);
        dset_id = H5Dcreate2(file_id, "data", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, pattern);
        H5Dclose(dset_id);
        if (0 == nn || 1 == nn || 2 == nn)
        {
            dset_id = H5Dcreate2(file_id, "phase", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, phase);
            H5Dclose(dset_id);
        }
        H5Sclose(dspace_id);

        dspace_id = H5Screate_simple(1, dims_angle, NULL);
        dset_id = H5Dcreate2(file_id, "angle", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rot_angle);
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
        H5Fclose(file_id);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == world_rank)
        printf("Gather angle...\n");
    MPI_Gather(pattern_angle_i, batch * 4, MPI_DOUBLE, pattern_angle, batch * 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (0 == world_rank)
    {
        snprintf(file_out, 400, "%s/angle.h5", output_path);

        hid_t file_id, dset_id, dspace_id;
        hsize_t dims[2] = {batch * world_size, 4};
        herr_t status;

        file_id = H5Fcreate(file_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        dspace_id = H5Screate_simple(2, dims, NULL);
        dset_id = H5Dcreate2(file_id, "angle", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, pattern_angle);
        H5Dclose(dset_id);
        H5Sclose(dspace_id);
        H5Fclose(file_id);
        printf("\nFinish ! ! !\n");
    }
    nfft_finalize(&p);
    fclose(fp);
    free(pattern_angle_i);
    free(pixels);
    free(pattern_angle);
    nfft_free(data);
    MPI_Finalize();

    return 0;
}

void GenPixels(double *pixels, double lambda, double z_det, double pix_len, int size)
{
    // 根据模拟衍射的各项参数，构造出每个像素点对应的坐标数组Pixels = {x1, y1, z1, omega1, x2, ..., }，以倒空间坐标原点为原点，向下为x轴正方形，向右为y轴正方向，建立右手系
    // 计算结果:探测器上x方向的最大分辨率,也是Volume中x方向最大的分辨率,Volume中y,z方向的最大分辨率与x相同.

    double x_plane, y_plane, z_plane, x_sphere, y_sphere, z_sphere; // plane代表探测器平面上像素点的坐标，sphere代表投影到Ewald球后曲面上像素点的坐标。
    double length;                                                  // 以正空间样品位置为原点的坐标, 探测器与样品之间的距离length.
    double pixels0;                                                 // 每个方向上最边缘像素点对应的真实长度.
    double omega, cos_2theta;                                       // 立体角omega，衍射角2theta
    long i, j, t;
    double center, dsize;
    dsize = size;             // 将size转化为double类型
    center = (dsize - 1) / 2; // 中心位置

    // 计算出pixels0，物理含义为x=0，y=center探测器平面上的点，变换到Ewald球后得到的x坐标。
    i = 0;
    t = 0;
    x_plane = (i - center) * pix_len;
    y_plane = 0;
    z_plane = -z_det; // 为了满足建立的右手坐标系，z值需要取负
    length = sqrt(x_plane * x_plane + y_plane * y_plane + z_plane * z_plane);
    x_sphere = x_plane / (length * lambda); // 计算中运用了近似:sin(2*theta)=2sin(theta)
    pixels0 = fabs(x_sphere);

    // 计算pixels[]，坐标参数缩放到（-center，center）范围
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            t = i * size + j;
            x_plane = (i - center) * pix_len;
            y_plane = (j - center) * pix_len;
            z_plane = -z_det;
            length = sqrt(x_plane * x_plane + y_plane * y_plane + z_plane * z_plane);
            x_sphere = x_plane / (length * lambda);
            y_sphere = y_plane / (length * lambda);
            z_sphere = z_plane / (length * lambda);
            cos_2theta = z_det / length;
            omega = (pix_len * pix_len) * cos_2theta / (length * length);
            pixels[t * 4] = center * x_sphere / pixels0;
            pixels[t * 4 + 1] = center * y_sphere / pixels0;
            pixels[t * 4 + 2] = center * (z_sphere + 1 / lambda) / pixels0;
            pixels[t * 4 + 3] = omega;
        }
    }
}

void MakeRotMatrixEuler(double *rot_angle, double rot_matrix[3][3])
{
    // 输入旋转角度rot_angle, 计算出旋转矩阵rot_matrix
    //  rot_angle={alpha, beta, gamma}：以入射X射线方向为z轴负方向，建立右手坐标系，rot_angle表示这个坐标系相对于样品颗粒坐标系的旋转角度，三个角度依次对应于zxz顺规下的欧拉角。
    //  rot_mtrix：由于输入的旋转角是X射线的欧拉角，而rot_matrix代表样品的旋转矩阵，其数值上等于输入欧拉角对应旋转矩阵的逆矩阵。
    double alpha, beta, gamma;
    double s1, c1, s2, c2, s3, c3;

    alpha = rot_angle[0];
    beta = rot_angle[1];
    gamma = rot_angle[2];
    s1 = sin(alpha);
    c1 = cos(alpha);
    s2 = sin(beta);
    c2 = cos(beta);
    s3 = sin(gamma);
    c3 = cos(gamma);

    rot_matrix[0][0] = -s1 * c2 * s3 + c3 * c1;
    rot_matrix[0][1] = -s1 * c2 * c3 - s3 * c1;
    rot_matrix[0][2] = s1 * s2;
    rot_matrix[1][0] = c1 * c2 * s3 + c3 * s1;
    rot_matrix[1][1] = c1 * c2 * c3 - s3 * s1;
    rot_matrix[1][2] = -c1 * s2;
    rot_matrix[2][0] = s2 * s3;
    rot_matrix[2][1] = s2 * c3;
    rot_matrix[2][2] = c2;
}

void RotatePixels(double *pixels, double rot_matrix[3][3], double rot_pix[3], int size)
{
    long t, i, j;
    double center, dsize;
    dsize = size;             // 将size转化为double类型
    center = (dsize - 1) / 2; // 中心位置

    for (i = 0; i < 3; i++)
    {
        rot_pix[i] = 0;
        for (j = 0; j < 3; j++)
        {
            rot_pix[i] += rot_matrix[i][j] * pixels[t * 4 + j];
        }
    }
}

int min(int x, int y)
{
    int z;
    if (x > y)
    {
        z = y;
    }
    else
    {
        z = x;
    }
    return (z);
}
