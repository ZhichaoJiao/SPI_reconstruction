/*
Writen by Zhi Geng and Zhichao Jiao
*/

#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <string.h>
#include <getopt.h>

int main(int argc, char *argv[])
{
    int size = 512;
    int bin = 2;      // 首先进行bin操作
    int resize = 256; // 然后截取中央resize大小的范围

    int resize_volume = 0;
    char input_volume_path[400] = "/jiao/Data/6ZFP/6zfp_pattern_1_256/volume_6zfp_af2.h5";
    char output_volume_path[400] = "/jiao/Data/6ZFP/6zfp_pattern_1_128/volume_6zfp_af2.h5";

    int resize_pattern = 0;
    char input_pattern_path[400] = "/jiao/Data/6ZFP/6zfp_pattern_1_256/pattern_";
    char output_pattern_path[400] = "/jiao/Data/6ZFP/6zfp_pattern_1_128/pattern_";
    int n_pattern = 10000;

    /*Long options*/
    const struct option longopts[] = {
        {"resize_volume", 1, NULL, 1},
        {"input_volume_path", 1, NULL, 2},
        {"output_volume_path", 1, NULL, 3},
        {"resize_pattern", 1, NULL, 4},
        {"input_pattern_path", 1, NULL, 5},
        {"output_pattern_path", 1, NULL, 6},
        {"n_pattern", 1, NULL, 7},
        {"size", 1, NULL, 8},
        {"resize", 1, NULL, 9},
        {"bin", 1, NULL, 10},

        {0, 0, NULL, 0}};

    /*short options*/
    /*从命令中读取参数信息，如果有输入，那么将输入赋给参数，如果没有输入，那么使用参数变量声明时候的初始值*/
    int c;
    char *rval;

    while ((c = getopt_long(argc, argv, "i:", longopts, NULL)) != -1)
    {
        switch (c)
        {
        case 1:
            resize_volume = strtol(optarg, &rval, 10);
            break;

        case 2:
            snprintf(input_volume_path, 400, "%s", optarg);
            break;

        case 3:
            snprintf(output_volume_path, 400, "%s", optarg);
            break;

        case 4:
            resize_pattern = strtol(optarg, &rval, 10);
            break;

        case 5:
            snprintf(input_pattern_path, 400, "%s", optarg);
            break;

        case 6:
            snprintf(output_pattern_path, 400, "%s", optarg);
            break;

        case 7:
            n_pattern = strtol(optarg, &rval, 10);
            break;

        case 8:
            size = strtol(optarg, &rval, 10);
            break;

        case 9:
            resize = strtol(optarg, &rval, 10);
            break;

        case 10:
            bin = strtol(optarg, &rval, 10);
            break;
        }
    }

    printf("size = %d\n", size);
    printf("bin = %d\n",bin);
    printf("resize = %d\n", resize);
    printf("n_pattern = %d\n", n_pattern);
    printf("resize_volume = %d\n", resize_volume);
    printf("input_volume_path = %s\n", input_volume_path);
    printf("output_volume_path = %s\n", output_volume_path);
    printf("resize_pattern = %d\n", resize_pattern);
    printf("input_pattern_path = %s\n", input_pattern_path);
    printf("output_pattern_path = %s\n", output_pattern_path);

    int binsize, low; // bin操作后的size
    if (0 == bin)
        binsize = size;
    else
        binsize = size / bin;
    low = (binsize - resize) / 2;

    if (resize_volume)
    {
        float(*volume_in)[size][size] = (float(*)[size][size])malloc(sizeof(float) * size * size * size); // 三维衍射强度三维数组
        memset(volume_in, 0, sizeof(float) * size * size * size);
        float(*volume_bin)[binsize][binsize] = (float(*)[binsize][binsize])malloc(sizeof(float) * binsize * binsize * binsize); // 三维衍射强度三维数组
        memset(volume_bin, 0, sizeof(float) * binsize * binsize * binsize);
        float(*volume_out)[resize][resize] = (float(*)[resize][resize])malloc(sizeof(float) * resize * resize * resize); // 三维衍射强度三维数组
        memset(volume_out, 0, sizeof(float) * resize * resize * resize);

        // 输入初始volume
        hid_t file_volume_in, dataset, dataspace;
        herr_t status;
        file_volume_in = H5Fopen(input_volume_path, H5F_ACC_RDONLY, H5P_DEFAULT);
        dataset = H5Dopen(file_volume_in, "volume_pow", H5P_DEFAULT);
        status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, volume_in);
        H5Dclose(dataset);
        status = H5Fclose(file_volume_in);

        // 压缩尺寸
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                for (int k = 0; k < size; k++)
                {
                    int ii, jj, kk;
                    ii = i / bin;
                    jj = j / bin;
                    kk = k / bin;
                    volume_bin[ii][jj][kk] += volume_in[i][j][k];
                }

        // 截取中心部分
        for (int i = 0; i < resize; i++)
            for (int j = 0; j < resize; j++)
                for (int k = 0; k < resize; k++)
                {
                    int ii, jj, kk;
                    ii = low + i;
                    jj = low + j;
                    kk = low + k;
                    volume_out[i][j][k] = volume_bin[ii][jj][kk];
                }

        // 输出压缩后的volume
        hid_t file_volume_out;
        file_volume_out = H5Fcreate(output_volume_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dim_3[3] = {resize, resize, resize};
        dataspace = H5Screate_simple(3, dim_3, NULL);
        dataset = H5Dcreate(file_volume_out, "volume_pow", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, volume_out);
        H5Dclose(dataset);
        H5Sclose(dataspace);
        status = H5Fclose(file_volume_out);

        free(volume_in);
        free(volume_bin);
        free(volume_out);
    }

    if (resize_pattern)
    {
        for (int index_pattern = 0; index_pattern < n_pattern; index_pattern++)
        {
            char input_pattern_path_i[400] = {};
            char output_pattern_path_i[400] = {};
            float pattern_in[size][size];
            memset(pattern_in, 0, sizeof(float) * size * size);
            float pattern_bin[binsize][binsize];
            memset(pattern_bin, 0, sizeof(float) * binsize * binsize);
            float pattern_out[resize][resize];
            memset(pattern_out, 0, sizeof(float) * resize * resize);

            // 输入初始pattern
            snprintf(input_pattern_path_i, 400, "%s%d.h5", input_pattern_path, index_pattern);
            hid_t file_pattern_in, dataset, dataspace;
            herr_t status;
            file_pattern_in = H5Fopen(input_pattern_path_i, H5F_ACC_RDONLY, H5P_DEFAULT);
            dataset = H5Dopen(file_pattern_in, "data", H5P_DEFAULT);
            status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, pattern_in);
            H5Dclose(dataset);
            status = H5Fclose(file_pattern_in);

            // 压缩尺寸
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                {
                    int ii, jj, kk;
                    ii = i / bin;
                    jj = j / bin;
                    pattern_bin[ii][jj] += pattern_in[i][j];
                }
            
            for (int i = 0; i < resize; i++)
                for (int j = 0; j < resize; j++)
                {
                    int ii, jj;
                    ii = low + i;
                    jj = low + j;
                    pattern_out[i][j] = pattern_bin[ii][jj];
                }

            // 输出压缩后的pattern
            snprintf(output_pattern_path_i, 400, "%s%d.h5", output_pattern_path, index_pattern);
            hid_t file_pattern_out;
            file_pattern_out = H5Fcreate(output_pattern_path_i, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hsize_t dim_3[2] = {resize, resize};
            dataspace = H5Screate_simple(2, dim_3, NULL);
            dataset = H5Dcreate(file_pattern_out, "data", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Dwrite(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, pattern_out);
            H5Dclose(dataset);
            H5Sclose(dataspace);
            status = H5Fclose(file_pattern_out);
        }
    }

    return 0;
}
