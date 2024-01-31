/*
Writen by Zhi Geng and Zhichao Jiao
*/

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdarg.h>
#include <getopt.h>
#include <time.h>
#include <fftw3.h>
#include <omp.h>

#define pi 3.1415926
void ifftshift(int i, int j, int k, int size, int *ii, int *jj, int *kk);

int main(int argc, char *argv[])
{
    printf("Phasing begin...\n");
    // 声明可调变量
    int size = 0;         // 输入的volume大小
    int n_bin = 1;        // 将volume边长缩小n_bin倍
    int beam_stop = 0;    // beam_stop边长的一半,按照bin前计算,单位为像素数量
    int support_size = 0; // 正空间support大小
    double beta = 0;      // HIO算法中的参数beta
    int input_phase = 0;  // 是否输入初始相位,如果为0代表使用随机初始相位
    int n_hio, n_er;      // hio与er算法的迭代次数,先hio后er
    char volume_file[400] = "";
    char phase_file[400] = "";
    char output_path[400] = "";

    int c;
    char *rval;
    const struct option longopts[] = {
        {"help", 0, NULL, 'h'},
        {"size", 1, NULL, 0},
        {"n_bin", 1, NULL, 1},
        {"beam_stop", 1, NULL, 2},
        {"support_size", 1, NULL, 3},
        {"n_hio", 1, NULL, 5},
        {"n_er", 1, NULL, 6},
        {"beta", 1, NULL, 7},
        {"input_phase", 1, NULL, 8},
        {"phase_file", 1, NULL, 9},
        {"volume_file", 1, NULL, 10},
        {"output_path", 1, NULL, 11},

        {0, 0, NULL, 0}};

    while ((c = getopt_long(argc, argv, "h", longopts, NULL)) != -1)
    {

        switch (c)
        {
        case 0:
            size = strtol(optarg, &rval, 10);
            break;

        case 1:
            n_bin = strtol(optarg, &rval, 10);
            break;

        case 2:
            beam_stop = strtol(optarg, &rval, 10);
            break;

        case 3:
            support_size = strtol(optarg, &rval, 10);
            break;

        case 5:
            n_hio = strtol(optarg, &rval, 10);
            break;

        case 6:
            n_er = strtol(optarg, &rval, 10);
            break;

        case 7:
            beta = strtod(optarg, &rval);
            break;

        case 8:
            input_phase = strtol(optarg, &rval, 10);
            break;

        case 9:
            snprintf(phase_file, 400, "%s", optarg);
            break;

        case 10:
            snprintf(volume_file, 400, "%s", optarg);
            break;

        case 11:
            snprintf(output_path, 400, "%s", optarg);
            break;
        }
    }

    // 声明其它变量
    int resize = size / n_bin;
    fftw_plan p;                                                                                      // bin2后的volume大小
    float(*intensity)[size][size] = (float(*)[size][size])malloc(sizeof(float) * size * size * size); // 倒空间强度数组
    memset(intensity, 0, sizeof(float) * size * size * size);
    float(*phase)[size][size] = (float(*)[size][size])malloc(sizeof(float) * size * size * size); // 倒空间相位数组
    memset(phase, 0, sizeof(float) * size * size * size);
    float *fft_out_shift = (float *)malloc(size * size * size * sizeof(float)); // 傅立叶变换输出实部一维数组
    memset(fft_out_shift, 0, sizeof(float) * size * size * size);
    fftw_complex *volume = (fftw_complex *)fftw_malloc(size * size * size * sizeof(fftw_complex)); // 倒空间复数一维数组
    memset(volume, 0, sizeof(fftw_complex) * size * size * size);
    fftw_complex *fft_out = (fftw_complex *)fftw_malloc(size * size * size * sizeof(fftw_complex)); // 傅立叶变换输出复数一维数组
    memset(fft_out, 0, sizeof(fftw_complex) * size * size * size);

    float(*amplitude_resize)[resize][resize] = (float(*)[resize][resize])malloc(sizeof(float) * resize * resize * resize); // 倒空间振幅数组,bin后
    memset(amplitude_resize, 0, sizeof(float) * resize * resize * resize);
    float *amplitude_resize_1d = (float *)malloc(sizeof(float) * resize * resize * resize);
    memset(amplitude_resize_1d, 0, sizeof(float) * resize * resize * resize);
    float(*density_resize)[resize][resize] = (float(*)[resize][resize])malloc(sizeof(float) * resize * resize * resize); // 正空间电子密度数组,bin后
    memset(density_resize, 0, sizeof(float) * resize * resize * resize);
    float(*density_support)[support_size][support_size] = (float(*)[support_size][support_size])malloc(sizeof(float) * support_size * support_size * support_size); // 正空间电子密度数组,bin后
    memset(density_support, 0, sizeof(float) * support_size * support_size * support_size);
    float *density_resize_1d_last = (float *)malloc(sizeof(float) * resize * resize * resize);
    memset(density_resize_1d_last, 0, sizeof(float) * resize * resize * resize);
    int *support = (int *)malloc(resize * resize * resize * sizeof(int)); // 正空间约束区域
    memset(support, 0, sizeof(int) * resize * resize * resize);
    int *mask = (int *)malloc(resize * resize * resize * sizeof(int)); // 倒空间beam_mask区域
    memset(mask, 0, sizeof(int) * resize * resize * resize);

    char log_file[400], parameter_file[400];
    snprintf(log_file, 400, "%s/log.txt", output_path);
    FILE *log = fopen(log_file, "w"); // 输出R因子等参数
    // fprintf(log, "%-*s%-*s%-*s%-*s%-*s%-*s\n", 12, "Iteration", 12, "R Factor", 12, "R Factor(norm)  ", 12, "F(0,0,0)", 12, "CC", 12, "Gamma");
    fprintf(log, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "Iteration", "R Factor", "R Factor(norm)", "F(0,0,0)", "CC", "Gamma", "Density_change");

    fclose(log);
    snprintf(parameter_file, 400, "%s/parameter.txt", output_path);
    FILE *parameter = fopen(parameter_file, "w"); // 输出参数
    fprintf(parameter, "size: %d\n", size);
    fprintf(parameter, "beam_stop: %d\n", beam_stop);
    fprintf(parameter, "n_bin: %d\n", n_bin);
    fprintf(parameter, "support_size: %d\n", support_size);
    fprintf(parameter, "n_hio: %d\n", n_hio);
    fprintf(parameter, "n_er: %d\n", n_er);
    fprintf(parameter, "beta: %f\n", beta);
    fprintf(parameter, "input_phase: %d\n", input_phase);
    fprintf(parameter, "volume_file: %s\n", volume_file);
    fprintf(parameter, "phase_file: %s\n", phase_file);
    fprintf(parameter, "ouput_path: %s\n", output_path);
    fclose(parameter);

    // 读取衍射强度
    hid_t file_id, data_id;
    herr_t status;
    file_id = H5Fopen(volume_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    data_id = H5Dopen2(file_id, "volume_pow", H5P_DEFAULT);
    status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, intensity);
    H5Dclose(data_id);
    status = H5Fclose(file_id);

    // 生成初始相位
    if (1 == input_phase)
    {
        float(*amplitude_af2)[size][size] = (float(*)[size][size])malloc(sizeof(float) * size * size * size); // 倒空间强度数组
        memset(amplitude_af2, 0, sizeof(float) * size * size * size);
        file_id = H5Fopen(phase_file, H5F_ACC_RDONLY, H5P_DEFAULT);
        data_id = H5Dopen2(file_id, "phase", H5P_DEFAULT);
        status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, phase);
        H5Dclose(data_id);
        data_id = H5Dopen2(file_id, "volume_pow", H5P_DEFAULT);
        status = H5Dread(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, amplitude_af2);
        H5Dclose(data_id);
        status = H5Fclose(file_id);
        // 合并振幅与相位
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; k++)
                {
                    int ii, jj, kk, nn;
                    ifftshift(i, j, k, size, &ii, &jj, &kk);
                    nn = ii * size * size + jj * size + kk;
                    volume[nn] = sqrt(amplitude_af2[i][j][k]) * cexp(I * phase[i][j][k]);
                }
    }
    else if (0 == input_phase)
    {
        srand(time(NULL));
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; k++)
                {
                    double rand_num = (double)rand() / RAND_MAX; // 生成0到1之间的随机数
                    phase[i][j][k] = 2 * pi * rand_num;          // 转换为0到2*pi之间的随机数
                }
        // 合并振幅与相位
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; k++)
                {
                    int ii, jj, kk, nn;
                    ifftshift(i, j, k, size, &ii, &jj, &kk);
                    nn = ii * size * size + jj * size + kk;
                    volume[nn] = sqrt(intensity[i][j][k]) * cexp(I * phase[i][j][k]);
                }
    }

    // 傅立叶变换
    p = fftw_plan_dft_3d(size, size, size, volume, fft_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    // fft_shift, 将蛋白质颗粒shift到中心
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
            {
                int ii, jj, kk, n, nn;
                ifftshift(i, j, k, size, &ii, &jj, &kk);
                n = i * size * size + j * size + k;
                nn = ii * size * size + jj * size + kk;
                fft_out_shift[nn] = cabs(fft_out[n]) / (resize * resize * resize);
            }

    fftw_free(volume);
    fftw_free(fft_out);

    fftw_complex *density_resize_1d = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * resize * resize * resize);
    memset(density_resize_1d, 0, sizeof(fftw_complex) * resize * resize * resize);
    fftw_complex *volume_resize = (fftw_complex *)fftw_malloc(resize * resize * resize * sizeof(fftw_complex));
    memset(volume_resize, 0, sizeof(fftw_complex) * resize * resize * resize);
    float *volume_resize_cabs = (float *)malloc(sizeof(float) * resize * resize * resize);
    memset(volume_resize_cabs, 0, sizeof(float) * resize * resize * resize);

    // 截取电子密度中心部分
    for (int i = 0; i < resize; i++)
        for (int j = 0; j < resize; j++)
            for (int k = 0; k < resize; k++)
            {
                int ii, jj, kk, n, nn, low;
                low = (size - resize) / 2;
                ii = i + low;
                jj = j + low;
                kk = k + low;
                n = i * resize * resize + j * resize + k;
                nn = ii * size * size + jj * size + kk;
                density_resize[i][j][k] = fft_out_shift[nn];
                density_resize_1d[n] = fft_out_shift[nn];
            }

    // 输出初始电子密度
    hid_t dataspace;
    char file_out[400] = {0};
    snprintf(file_out, 400, "%s/initial_density.h5", output_path);
    file_id = H5Fcreate(file_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dim_3[3] = {resize, resize, resize};
    dataspace = H5Screate_simple(3, dim_3, NULL);
    data_id = H5Dcreate(file_id, "density", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, density_resize);
    H5Dclose(data_id);
    H5Sclose(dataspace);
    H5Fclose(file_id);

    // bin:resize倒空间衍射强度
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
            {
                int ii, jj, kk;
                ii = i / n_bin;
                jj = j / n_bin;
                kk = k / n_bin;
                amplitude_resize[ii][jj][kk] += sqrt(intensity[i][j][k]);
            }

    // 使倒空间振幅满足Friedel定律
    for (int i = 1; i < resize; i++)
        for (int j = 1; j < resize; j++)
            for (int k = 1; k < resize / 2 + 1; k++)
            {
                amplitude_resize[i][j][k] = (amplitude_resize[i][j][k] + amplitude_resize[resize - i][resize - j][resize - k]) / 2;
                amplitude_resize[resize - i][resize - j][resize - k] = amplitude_resize[i][j][k];
            }
    for (int i = 1; i < resize; i++)
    {
        amplitude_resize[i][0][0] = (amplitude_resize[i][0][0] + amplitude_resize[resize - i][0][0]) / 2;
        amplitude_resize[resize - i][0][0] = amplitude_resize[i][0][0];
        for (int j = 1; j < resize / 2 + 1; j++)
        {
            amplitude_resize[i][j][0] = (amplitude_resize[i][j][0] + amplitude_resize[resize - i][resize - j][0]) / 2;
            amplitude_resize[resize - i][resize - j][0] = amplitude_resize[i][j][0];
        }
    }
    for (int j = 1; j < resize; j++)
    {
        amplitude_resize[0][j][0] = (amplitude_resize[0][j][0] + amplitude_resize[0][resize - j][0]) / 2;
        amplitude_resize[0][resize - j][0] = amplitude_resize[0][j][0];
        for (int k = 1; k < resize; k++)
        {
            amplitude_resize[0][j][k] = (amplitude_resize[0][j][k] + amplitude_resize[0][resize - j][resize - k]) / 2;
            amplitude_resize[0][resize - j][resize - k] = amplitude_resize[0][j][k];
        }
    }

    for (int k = 1; k < resize; k++)
    {
        amplitude_resize[0][0][k] = (amplitude_resize[0][0][k] + amplitude_resize[0][0][resize - k]) / 2;
        amplitude_resize[0][0][resize - k] = amplitude_resize[0][0][k];
        ;
        for (int i = 1; i < resize; i++)
        {
            amplitude_resize[i][0][k] = (amplitude_resize[i][0][k] + amplitude_resize[resize - i][0][resize - k]) / 2;
            amplitude_resize[resize - i][0][resize - k] = amplitude_resize[i][0][k];
        }
    }

    // fft_shift, 将倒空间衍射强度进行shift
    for (int i = 0; i < resize; i++)
        for (int j = 0; j < resize; j++)
            for (int k = 0; k < resize; k++)
            {
                int ii, jj, kk, nn;
                ifftshift(i, j, k, resize, &ii, &jj, &kk);
                nn = ii * resize * resize + jj * resize + kk;
                amplitude_resize_1d[nn] = amplitude_resize[i][j][k];
            }

    // 生成倒空间mask
    beam_stop = beam_stop / n_bin;
    for (int i = 0; i < beam_stop * 2 + 1; i++)
        for (int j = 0; j < beam_stop * 2 + 1; j++)
            for (int k = 0; k < beam_stop * 2 + 1; k++)
            {
                int ii, jj, kk, nn, low;
                low = (resize - beam_stop * 2) / 2;
                ifftshift(i + low, j + low, k + low, resize, &ii, &jj, &kk);
                nn = ii * resize * resize + jj * resize + kk;
                mask[nn] = 1;
            }
    /*for (int n = 0; n < resize * resize * resize; n++)
        if (amplitude_resize_1d[n] == 0)
            mask[n] = 1;*/

    // 生成正空间support
    int low = (resize - support_size) / 2;
    for (int i = 0; i < support_size; i++)
        for (int j = 0; j < support_size; j++)
            for (int k = 0; k < support_size; k++)
            {
                int ii, jj, kk, nn;
                ii = i + low;
                jj = j + low;
                kk = k + low;
                nn = ii * resize * resize + jj * resize + kk;
                support[nn] = 1;
            }

    free(intensity);
    free(phase);
    free(fft_out_shift);

    printf("%-*s%-*s%-*s%-*s%-*s%-*s%-*s\n", 8, "Iter", 13, "R Factor", 14, "R Factor(norm)", 13, "F(0,0,0)", 13, "CC", 13, "Gamma", 13, "Den_change");
    // 从初始电子密度图开始,进行双空间迭代
    for (int i_iteration = 1; i_iteration < n_hio + n_er+1; i_iteration++)
    {
#pragma omp parallel for
        for (int n = 0; n < resize * resize * resize; n++)
            density_resize_1d_last[n] = creal(density_resize_1d[n]);

        // 变换到倒空间
        p = fftw_plan_dft_3d(resize, resize, resize, density_resize_1d, volume_resize, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);

#pragma omp parallel for
        for (int n = 0; n < resize * resize * resize; n++)
            volume_resize_cabs[n] = cabs(volume_resize[n]);

        // 计算R因子
        double f_000, sum1, sum2, k_up_sum, k_down_sum, fc;
        f_000 = cabs(volume_resize[0]);
        sum1 = 0;
        sum2 = 0;
        for (int n = 0; n < resize * resize * resize; n++)
        {
            if (mask[n] == 1)
                continue;

            fc = volume_resize_cabs[n];
            sum1 += fabs(fc - amplitude_resize_1d[n]);
            sum2 += amplitude_resize_1d[n];

            // 计算归一化R因子
            k_up_sum += fc * amplitude_resize_1d[n];
            k_down_sum += fc * fc;
        }
        double R_factor = sum1 / sum2;

        // 计算归一化R因子
        double k_factor = k_up_sum / k_down_sum;
        sum1 = 0;
        for (int n = 0; n < resize * resize * resize; n++)
        {
            if (mask[n] == 1)
                continue;
            fc = volume_resize_cabs[n];
            sum1 += fabs(fc * k_factor - amplitude_resize_1d[n]);
        }
        double R_factor_norm = sum1 / sum2;

        // 计算倒空间cc系数
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
        double sum_x2 = 0.0, sum_y2 = 0.0;
        double cc;
        int n = resize * resize * resize;

        for (int i = 0; i < n; i++)
        {
            if (1 == mask[i])
                continue;
            sum_x += volume_resize_cabs[i];
            sum_y += amplitude_resize_1d[i];
            sum_xy += volume_resize_cabs[i] * amplitude_resize_1d[i];
            sum_x2 += volume_resize_cabs[i] * volume_resize_cabs[i];
            sum_y2 += amplitude_resize_1d[i] * amplitude_resize_1d[i];
        }
        cc = (n * sum_xy - sum_x * sum_y) / sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

        // 倒空间约束
        //int n = resize * resize * resize;
        sum_x = 0.0; 
        sum_y = 0.0;
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            if (1 == mask[i])
                continue;
            sum_x += volume_resize_cabs[i];
            sum_y += amplitude_resize_1d[i];
        }
        for (int n = 0; n < resize * resize * resize; n++)
        {
            if (1 == mask[n])
            {
                double phase_n = atan2(cimag(volume_resize[n]), creal(volume_resize[n]));
                volume_resize[n] = (volume_resize_cabs[n] * sum_y / sum_x) * cexp(I * phase_n);
                continue;
            }
            double phase_n = atan2(cimag(volume_resize[n]), creal(volume_resize[n]));
            volume_resize[n] = amplitude_resize_1d[n] * cexp(I * phase_n);
        }

        // 变换到正空间
        p = fftw_plan_dft_3d(resize, resize, resize, volume_resize, density_resize_1d, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p);

        // 计算g_factor
        double g_factor = 0, sum_up = 0, sum_down = 0;
        int n_up = 0, n_down = 0;

        for (int n = 0; n < resize * resize * resize; n++)
        {
            if (1 == support[n])
            {
                sum_down += cabs(density_resize_1d[n]);
                n_down += 1;
            }
            else if (0 == support[n])
            {
                sum_up += cabs(density_resize_1d[n]);
                n_up += 1;
            }
        }
        g_factor = (sum_up * n_down) / (sum_down * n_up);

// 正空间约束
#pragma opm parallel for
        for (int n = 0; n < resize * resize * resize; n++)
        {
            // 检查正空间相位是否为0
            double phase_test = atan2(cimag(density_resize_1d[n]), creal(density_resize_1d[n]));
            if (!(fabs(phase_test) < 0.1 || fabs(phase_test - pi) < 0.1 || fabs(phase_test + pi) < 0.1))
            {
                printf("Error: density[%d] is not a real number, phase[%d] = %f\n", n, n, phase_test);
                return 1;
            }
            density_resize_1d[n] = creal(density_resize_1d[n]) / (resize * resize * resize);
            if (1 == support[n])
            {
                if (creal(density_resize_1d[n]) < 0)
                {
                    if (i_iteration < n_hio)
                        density_resize_1d[n] = density_resize_1d_last[n] - beta * creal(density_resize_1d[n]);
                    else
                        density_resize_1d[n] = 0;
                }
            }
            else if (0 == support[n])
            {
                if (i_iteration < n_hio)
                    density_resize_1d[n] = density_resize_1d_last[n] - beta * creal(density_resize_1d[n]);
                else
                    density_resize_1d[n] = 0;
            }
        }

        // 计算电子密度变化大小density_change
        double density_change = 0, density_change_up = 0, density_change_down = 0;
        for (int n = 0; n < resize * resize * resize; n++)
        {
            if (1 == support[n])
            {
                density_change_up += fabs(creal(density_resize_1d[n]) - density_resize_1d_last[n]);
                density_change_down += fabs(creal(density_resize_1d[n]));
            }
        }
        density_change = density_change_up / density_change_down;

        printf("%-*d%-*f%-*f%-*d%-*f%-*f%-*f\n", 8, i_iteration, 13, R_factor, 14, R_factor_norm, 13, (int)f_000, 13, cc, 13, g_factor, 13, density_change);
        log = fopen(log_file, "a");
        // fprintf(log, "%-*d%-*f%-*f%-*d%-*f%-*f\n", 16, i_iteration, 16, R_factor, 16, R_factor_norm, 16, (int)f_000, 16, cc, 16, g_factor);
        fprintf(log, "%d\t%f\t%f\t%f\t%f\t%f\t%f\n", i_iteration, R_factor, R_factor_norm, round(f_000), cc, g_factor, density_change);
        fclose(log);

        // 将结果转化成三维
        if (i_iteration % 20 == 0||i_iteration==1)
        {
            int low = (resize - support_size) / 2;
#pragma omp parallel for
            for (int i = 0; i < support_size; i++)
                for (int j = 0; j < support_size; j++)
                    for (int k = 0; k < support_size; k++)
                    {
                        int ii, jj, kk, n;
                        ii = i + low;
                        jj = j + low;
                        kk = k + low;
                        n = ii * resize * resize + jj * resize + kk;
                        density_support[i][j][k] = creal(density_resize_1d[n]);
                    }
            snprintf(file_out, 400, "%s/density_%d.h5", output_path, i_iteration);
            file_id = H5Fcreate(file_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hsize_t dim_2[3] = {support_size, support_size, support_size};
            dataspace = H5Screate_simple(3, dim_2, NULL);
            data_id = H5Dcreate(file_id, "density", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Dwrite(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, density_support);
            H5Dclose(data_id);
            H5Sclose(dataspace);
            H5Fclose(file_id);
        }
    }

    for (int i = 0; i < resize; i++)
        for (int j = 0; j < resize; j++)
            for (int k = 0; k < resize; k++)
            {
                int n;
                n = i * resize * resize + j * resize + k;
                density_resize[i][j][k] = creal(density_resize_1d[n]);
            }
    snprintf(file_out, 400, "%s/result_density.h5", output_path);
    file_id = H5Fcreate(file_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dim_2[3] = {resize, resize, resize};
    dataspace = H5Screate_simple(3, dim_2, NULL);
    data_id = H5Dcreate(file_id, "density", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, density_resize);
    H5Dclose(data_id);
    H5Sclose(dataspace);
    H5Fclose(file_id);

    free(amplitude_resize);
    free(amplitude_resize_1d);
    free(density_support);
    free(density_resize);
    free(density_resize_1d_last);
    free(support);
    free(mask);
    fftw_free(density_resize_1d);
    fftw_free(volume_resize);
    free(volume_resize_cabs);
    return 0;
}

void ifftshift(int i, int j, int k, int size, int *ii, int *jj, int *kk)
{
    if (i < (size + 1) / 2)
        *ii = i + size / 2;
    else if (size % 2 == 0)
        *ii = i - size / 2;
    else
        *ii = i - size / 2 - 1;

    if (j < (size + 1) / 2)
        *jj = j + size / 2;
    else if (size % 2 == 0)
        *jj = j - size / 2;
    else
        *jj = j - size / 2 - 1;

    if (k < (size + 1) / 2)
        *kk = k + size / 2;
    else if (size % 2 == 0)
        *kk = k - size / 2;
    else
        *kk = k - size / 2 - 1;
}