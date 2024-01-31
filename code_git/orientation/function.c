/*
Writen by Zhi Geng and Zhichao Jiao
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fftw3.h>
#include <string.h>
#include <hdf5.h>
#include <time.h>
#define pi 3.141592653590

extern int size;
extern int n_gamma;
extern int n_r;

void PatternCartFloat2Double(float pattern_cart_float[size][size], double pattern_cart_double[size][size])
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			pattern_cart_double[i][j] = pattern_cart_float[i][j];
		}
	}
}

void Volume3dFloat2Double(float volume_3d_float[size][size][size], double volume_3d_double[size][size][size])
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				volume_3d_double[i][j][k] = volume_3d_float[i][j][k];
			}
		}
	}
}

void Volume3dDouble2FLoat(double volume_3d_double[size][size][size], float volume_3d_float[size][size][size])
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				volume_3d_float[i][j][k] = volume_3d_double[i][j][k];
			}
		}
	}
}

void Volume_3d2Volume_1d(double volume_3d[size][size][size], float *volume_1d)
{
	memset(volume_1d, 0, sizeof(float) * size * size * size);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				long t = i * size * size + j * size + k;
				volume_1d[t] = volume_3d[i][j][k];
			}
		}
	}
}

void Volume_1d2Volume_3d(double volume_3d[size][size][size], float *volume_1d, double *weight, int size)
{
	// 根据Merge后的Volume_1d和权重weight计算出三维Volume分布
	memset(volume_3d, 0, sizeof(double) * size * size * size);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			for (int k = 0; k < size; k++)
			{
				long t = i * size * size + j * size + k;
				if (0 == volume_1d[t])
				{
					volume_3d[i][j][k] = volume_1d[t];
				}
				else
				{
					volume_3d[i][j][k] = volume_1d[t] / weight[t];
				}
			}
		}
	}
}

void Normalize(double polar[][n_gamma], int n_r, int n_gamma)
{
	// 输入一个极坐标的衍射图,将衍射图强度按照角度归一化,使得每一个r对应的所有角度的强度均值变成0,方差变成1.
	double mean, variance;

	for (int i = 0; i < n_r; i++)
	{
		mean = 0;
		variance = 0;

		for (int j = 0; j < n_gamma; j++)
			mean += polar[i][j] / n_gamma;

		for (int j = 0; j < n_gamma; j++)
			variance += pow(polar[i][j] - mean, 2) / n_gamma;

		for (int j = 0; j < n_gamma; j++)
		{
			if (0 == variance)
			{
				polar[i][j] = polar[i][j] - mean;
			}
			else
			{
				polar[i][j] = (polar[i][j] - mean) / sqrt(variance);
			}
		}
	}
}

void CorrelationCoefficientPolar_fft_many(const double pattern_polar[][n_gamma], const double reference_polar[][n_gamma], double *correlation_coefficient,
										  int n_r, int n_gamma, int r_min, int r_max)
{
	// 输入两张极坐标的衍射图,使用快速傅立叶变换计算相关系数,输出一个cc-gamma分布.
	// 在角度gamma维度对两个衍射图进行一维fft,然后将结果相乘再逆变换回来,最后将不同r的cc数据进行叠加.
	// 对于不同的r,对cc系数赋值不同的权重,权重大小为r.
	fftw_complex *fft_in, *fft_out_pattern, *fft_out_reference, *fft_out_cc;
	fftw_plan p;
	int t;
	int n[] = {n_gamma}; // 一维傅立叶变换的长度
	int r_sum = 0;		 // 所有r的权重之和 r_sum = r_min + (rmin+1) + ... + r_max.
	double(*pattern_polar_normalize)[n_gamma] = (double(*)[n_gamma])malloc(sizeof(double) * n_r * n_gamma);
	memset(pattern_polar_normalize, 0, sizeof(double) * n_r * n_gamma);
	double(*reference_polar_normalize)[n_gamma] = (double(*)[n_gamma])malloc(sizeof(double) * n_r * n_gamma);
	memset(reference_polar_normalize, 0, sizeof(double) * n_r * n_gamma);

	// Normalize(pattern_polar, pattern_polar_normalize, n_r, n_gamma);
	// Normalize(reference_polar, reference_polar_normalize, n_r, n_gamma);

	fft_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_gamma * n_r);
	fft_out_pattern = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_gamma * n_r);
	fft_out_reference = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_gamma * n_r);
	fft_out_cc = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n_gamma * n_r);

	// 对pattern做FFT

	for (int i = 0; i < n_r; i++)
	{
		for (int j = 0; j < n_gamma; j++)
		{
			t = i * n_gamma + j;
			fft_in[t][0] = pattern_polar[i][j];
			fft_in[t][1] = 0;
		}
	}

	p = fftw_plan_many_dft(1, n, n_r, fft_in, n, 1, n_gamma, fft_out_pattern, n, 1, n_gamma, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);

	// 对reference做FFT
	for (int i = 0; i < n_r; i++)
	{
		for (int j = 0; j < n_gamma; j++)
		{
			t = i * n_gamma + j;
			fft_in[t][0] = reference_polar[i][j];
			fft_in[t][1] = 0;
		}
	}

	p = fftw_plan_many_dft(1, n, n_r, fft_in, n, 1, n_gamma, fft_out_reference, n, 1, n_gamma, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);

	// 结果相乘,并作逆FFT
	for (int i = 0; i < n_gamma * n_r; i++)
	{
		fft_in[i][0] = fft_out_pattern[i][0] * fft_out_reference[i][0] + fft_out_pattern[i][1] * fft_out_reference[i][1];
		fft_in[i][1] = fft_out_pattern[i][1] * fft_out_reference[i][0] - fft_out_pattern[i][0] * fft_out_reference[i][1];
	}

	p = fftw_plan_many_dft(1, n, n_r, fft_in, n, 1, n_gamma, fft_out_cc, n, 1, n_gamma, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);

	// 计算cc
	// 计算权重
	for (int i = 0; i < n_gamma * n_r; i++)
	{
		fft_out_cc[i][0] /= n_gamma * n_gamma;
	}

	for (int j = 0; j < n_r; j++)
	{
		r_sum += r_min + j;
	}

	for (int i = 0; i < n_gamma; i++)
	{
		correlation_coefficient[n_gamma - i] = 0;
		for (int j = 0; j < n_r; j++)
		{
			t = j * n_gamma + i;
			//correlation_coefficient[n_gamma-i] += fft_out_cc[t][0] / n_r; //不赋权求均值
			correlation_coefficient[n_gamma - i] += fft_out_cc[t][0] * (r_min + j) / r_sum; // 赋权求均值
			if (fft_out_cc[t][1] > 0.0000001 && fft_out_cc[t][1] < -0.0000001)
			{
				printf("Error: cc is not a real number!!!\n fft_out_cc[%d][%d][1] = %f\n", i, j, fft_out_cc[t][1]);
			}
		}
	}

	fftw_free(fft_in);
	fftw_free(fft_out_pattern);
	fftw_free(fft_out_reference);
	fftw_free(fft_out_cc);
	free(pattern_polar_normalize);
	free(reference_polar_normalize);
}

void Cart2Polar(const double pattern_cart[size][size], double pattern_polar[][n_gamma], double r_min, double r_max, int n_gamma)
{
	// 将笛卡尔坐标系下的衍射数据pattern_cart转化成极坐标下表示pattern_polar,极坐标以x轴正方向为极轴正方向,逆时针方向为角度正方向.
	//  pattern_polar[r][gamma],极坐标下的衍射数据,其中r表示半径,gamma表示角度.
	// 输入r_min和r_max表示选取的半径最大值和最小值,也就是说其实是将直角坐标中的一个圆环变换成极坐标,半径的步长为1.
	//  n_gamma表示角度的采样个数,由此可以计算出角度的步长gamma_step.
	double r, gamma, gamma_step, x, y, fx, fy, cx, cy;
	double center, dsize;
	int tx, ty, n_r;

	dsize = size;			  // 将size转化为double类型
	center = (dsize - 1) / 2; // 中心位置
	n_r = r_max - r_min;
	gamma_step = 2 * pi / n_gamma;

	for (int r_index = 0; r_index < n_r; r_index++)
	{
		r = r_min + r_index;
		for (int gamma_index = 0; gamma_index < n_gamma; gamma_index++)
		{
			gamma = 0 + gamma_step * gamma_index;
			x = r * cos(gamma) + center;
			y = r * sin(gamma) + center;

			tx = x;
			ty = y;

			fx = x - tx;
			fy = y - ty;

			cx = 1. - fx;
			cy = 1. - fy;

			if (tx < 2 || tx > size - 2 || ty < 2 || ty > size - 2)
			{
				continue;
			}
			pattern_polar[r_index][gamma_index] = cx * cy * pattern_cart[tx][ty] + cx * fy * pattern_cart[tx][ty + 1] + fx * cy * pattern_cart[tx + 1][ty] + fx * fy * pattern_cart[tx + 1][ty + 1];
		}
	}
	Normalize(pattern_polar, n_r, n_gamma);
}

void FindBestAngle(const double pattern_polar[][n_gamma], const double reference_polar_matrix[][n_r][n_gamma], double rot_angle_matrix[][3],
				   double best_angle[3], int N_reference, double r_min, double r_max, int n_gamma, double dif_angle_cc[][4],
				   double *best_angle_cc, double *cc_max)
{ // dif_angle_cc[N_reference][4]记录了所有三维取向的cc,所有取向不包括自转角,记录下的cc为该取向下所有自转角的最大cc.
	// 每一行代表一个取向,前三个坐标对应欧拉角,最后一个坐标代表cc.
	// best_angle_cc[n_gamma]表示最佳取向下,所有自转角的cc系数

	double correlation_coefficient[n_gamma];
	double cc_max_allangle = 0, cc_max_1angle = 0; // cc_max_all代表所有取向中的cc最大值,cc_max_1angle代表一个确定的alpha, beta下所有自传角cc的最大值.
	int best_index = 0;
	for (int i = 0; i < N_reference; i++)
	{
		cc_max_1angle = 0;
		CorrelationCoefficientPolar_fft_many(pattern_polar, reference_polar_matrix[i], correlation_coefficient, n_r, n_gamma, r_min, r_max);

		// 找到每一个取向下最佳的自转角
		for (int j = 0; j < n_gamma; j++)
		{
			if (cc_max_1angle < correlation_coefficient[j])
			{
				cc_max_1angle = correlation_coefficient[j];
				dif_angle_cc[i][0] = rot_angle_matrix[i][0];
				dif_angle_cc[i][1] = rot_angle_matrix[i][1];
				dif_angle_cc[i][2] = j * 2 * pi / n_gamma;
				dif_angle_cc[i][3] = correlation_coefficient[j];
			}
		}

		// 判断是否是全空间最佳取向
		if (cc_max_allangle < cc_max_1angle)
		{
			cc_max_allangle = cc_max_1angle;
			best_angle[0] = dif_angle_cc[i][0];
			best_angle[1] = dif_angle_cc[i][1];
			best_angle[2] = dif_angle_cc[i][2];
			best_index = i;
			for (int k = 0; k < n_gamma; k++)
				best_angle_cc[k] = correlation_coefficient[k];
		}
	}
	*cc_max = cc_max_allangle;
}

void MakeAllAngle(double rot_angle_matrix[][3], double step, int *N_angle)
{
	// 根据所给步长step，计算出遍历空间的所有取向。
	//  step：遍历角度的步长，表示在大圆上两个相邻取向之间的角度间隔，注意在小圆上相邻取向之间的角度间隔更小，不同圆上的弧长间隔一致。单位是弧度，通常取0.03。
	//  rot_angle_matrix[][3]：记录空间取向的矩阵，每一行代表一个取向，三列依次代表zxz顺规下的三个欧拉角[alpha, beta, gamma]，单位是弧度。
	//  N_angle：记录采样的角度数量。
	int n_alpha, n_beta; // n_alpha代表每个确定β角下，α角度的取样个数。n_beta代表β角的取样个数。
	double alpha, beta;
	double step_alpha, step_beta; // α角度,β角度的取向间隔，单位是弧度。
	int t = 0;					  // 用于记录取向个数 m

	step_beta = step;
	n_beta = pi / step_beta; // 向下取整

	for (int i = 0; i <= n_beta; i++)
	{
		beta = i * step_beta;
		n_alpha = 2 * pi * sin(beta) / step_beta + 1; // 向上取整
		step_alpha = 2 * pi / n_alpha;
		for (int j = 0; j < n_alpha; j++)
		{
			alpha = j * step_alpha;
			rot_angle_matrix[t][0] = alpha;
			rot_angle_matrix[t][1] = beta;
			rot_angle_matrix[t][2] = 0.;
			t = t + 1;
		}
	}
	*N_angle = t;
}

void MakeFineAngle(double best_angle[3], double rot_angle_matrix[][3], double step, double step_fine, int fine_search)
{
	// 输入粗糙搜索后得到的最佳取向best_angle[3],粗糙搜索的步长step. 对最佳取向附近进行采样,采样范围附近两个步长,采样间隔step_fine. 输出采样角度.
	int n_alpha, n_beta; // n_alpha代表每个确定β角下，α角度的取样个数。n_beta代表β角的取样个数。
	double alpha, beta;
	double step_alpha, step_beta; // α角度,β角度的取向间隔，单位是弧度。
	int t = 0;					  // 用于记录取向个数
	double best_alpha, best_beta; // 记录粗糙搜索得到的最佳取向

	step_beta = step_fine ;
	n_beta = fine_search*2;
	best_alpha = best_angle[0];
	best_beta = best_angle[1];

	if (0 != best_beta)
	{
		n_alpha = 2 * pi * sin(best_beta) / step_beta + 1; // 向上取整
		step_alpha = 2 * pi / n_alpha;

		for (int i = 0; i < n_beta; i++)
		{
			beta = best_beta - step + i * step_beta;
			for (int j = 0; j < n_beta; j++)
			{
				alpha = best_alpha - step_alpha * fine_search + j * step_alpha;
				if(alpha<0){
					alpha = alpha + pi;
				}
				rot_angle_matrix[t][0] = alpha;
				rot_angle_matrix[t][1] = beta;
				rot_angle_matrix[t][2] = 0.;
				t = t + 1;
			}
		}
	}
	
	else if (0. == best_beta)
	{
		n_alpha = 2 * pi * sin(best_beta + step) / step_beta + 1; // 向上取整
		step_alpha = 2 * pi / n_alpha;
		for (int i = 0; i < n_beta; i++)
		{
			beta = best_beta + i * step_beta;
			for (int j = 0; j < n_beta; j++)
			{
				alpha = best_alpha - step_alpha * fine_search + j * step_alpha;
				if(alpha<0){
					alpha = alpha + pi;
				}
				rot_angle_matrix[t][0] = alpha;
				rot_angle_matrix[t][1] = beta;
				rot_angle_matrix[t][2] = 0.;
				t = t + 1;
			}
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

void GenPixels(double *pixels, double lambda, double z_det, double pix_len, int size)
{
	// 根据模拟衍射的各项参数，构造出每个像素点对应的坐标数组Pixels = {x1, y1, z1, omega1, x2, ..., }，以倒空间坐标原点为原点，向下为x轴正方形，向右为y轴正方向，建立右手系
	// 计算结果:探测器上x方向的最大分辨率,也是Volume中x方向最大的分辨率,Volume中y,z方向的最大分辨率与x相同.

	double x_plane, y_plane, z_plane, x_sphere, y_sphere, z_sphere; // plane代表探测器平面上像素点的坐标，sphere代表投影到Ewald球后曲面上像素点的坐标。
	double length;													// 以正空间样品位置为原点的坐标, 探测器与样品之间的距离length.
	double pixels0;													// 储存(x0^2+y0^2)^(1/2)，用于坐标的归一化
	double omega, cos_2theta;										// 立体角omega，衍射角2theta
	long i, j, t;
	double center, dsize;
	dsize = size;			  // 将size转化为double类型
	center = (dsize - 1) / 2; // 中心位置

	// 计算出pixels0，物理含义为x=0，y=center探测器平面上的点，变换到Ewald球后得到的x坐标。
	i = 0;
	t = 0;
	x_plane = (i - center) * pix_len;
	y_plane = 0;
	z_plane = -z_det; // 为了满足建立的右手坐标系，z值需要取负
	length = sqrt(x_plane * x_plane + y_plane * y_plane + z_plane * z_plane);
	x_sphere = x_plane / (length * lambda);
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

void ReferenceGen(double *rot_angle, double reference_cart[size][size], float *volume_1d, int size, double *pixels)
{
	// 输入一个取向rot_nagle,探测器每个像素点的坐标数组pixels,三维volume数据volume_3d,输出这个方向上的切片reference_cart.

	long t, i, j, x, y, z, N_pix = size * size;
	double tx, ty, tz, fx, fy, fz, cx, cy, cz;
	double rot_pix[3], rot_matrix[3][3];
	double *slice = (double *)malloc(sizeof(double) * size * size); // 直角坐标系下二维切片的一维数组
	memset(slice, 0, sizeof(double) * size * size);
	double center, dsize;
	dsize = size;			  // 将size转化为double类型
	center = (dsize - 1) / 2; // 中心位置

	MakeRotMatrixEuler(rot_angle, rot_matrix);

	for (t = 0; t < N_pix; t++)
	{
		for (i = 0; i < 3; i++)
		{
			rot_pix[i] = 0;
			for (j = 0; j < 3; j++)
			{
				rot_pix[i] += rot_matrix[i][j] * pixels[t * 4 + j];
			}
			rot_pix[i] += center;
		}
		tx = rot_pix[0];
		ty = rot_pix[1];
		tz = rot_pix[2];

		x = tx;
		y = ty;
		z = tz;

		if (x < 2 || x > size - 2 || y < 2 || y > size - 2 || z < 2 || z > size - 2)
		{
			slice[t] = 0.;
			continue;
		}

		fx = tx - x;
		fy = ty - y;
		fz = tz - z;
		cx = 1. - fx;
		cy = 1. - fy;
		cz = 1. - fz;

		slice[t] = cx * cy * cz * volume_1d[x * size * size + y * size + z] +
				   cx * cy * fz * volume_1d[x * size * size + y * size + ((z + 1) % size)] +
				   cx * fy * cz * volume_1d[x * size * size + ((y + 1) % size) * size + z] +
				   cx * fy * fz * volume_1d[x * size * size + ((y + 1) % size) * size + ((z + 1) % size)] +
				   fx * cy * cz * volume_1d[((x + 1) % size) * size * size + y * size + z] +
				   fx * cy * fz * volume_1d[((x + 1) % size) * size * size + y * size + ((z + 1) % size)] +
				   fx * fy * cz * volume_1d[((x + 1) % size) * size * size + ((y + 1) % size) * size + z] +
				   fx * fy * fz * volume_1d[((x + 1) % size) * size * size + ((y + 1) % size) * size + ((z + 1) % size)];
		slice[t] *= pixels[t * 4 + 3];
	}
	// 根据二维衍射图数据，将二维切片一维数组slice变成二维切片reference_cart
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			long t = i * size + j;
			reference_cart[i][j] = slice[t];
		}
	}
	free(slice);
}

void PatternMerge(double rot_angle[3], double pattern_cart[size][size], float *volume_1d, double *weight, int size, double *pixels)
{	
	// 根据衍射图的旋转角，将一张衍射图的强度插值到三维，得到Merge三维衍射强度的一维数组volume_1d和新的权重数组weight
	//  size表示每个维度（x/y）的像素点个数
	long x, y, z, N_pix = size * size;
	double tx, ty, tz, fx, fy, fz, cx, cy, cz, w, f;
	double rot_pix[3], rot_matrix[3][3] = {{0}};
	double *slice = (double *)malloc(sizeof(double) * size * size); // 直角坐标系下二维切片的一维数组
	memset(slice, 0, sizeof(double) * size * size);
	double center, dsize;
	dsize = size;			  // 将size转化为double类型
	center = (dsize - 1) / 2; // 中心位置

	// 构造旋转矩阵
	MakeRotMatrixEuler(rot_angle, rot_matrix);

	// 根据二维衍射图数据，将二维衍射强度pattern_cart变成一个一维数组slice
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			long t = i * size + j;
			slice[t] = pattern_cart[i][j];
		}
	}

	// 根据旋转矩阵，求出每个像素点旋转后的三维空间坐标，记录在rot_pix[3]中
	for (long t = 0; t < N_pix; t++)
	{
		for (int i = 0; i < 3; i++)
		{
			rot_pix[i] = 0.;
			for (int j = 0; j < 3; j++)
			{
				rot_pix[i] += rot_matrix[i][j] * pixels[t * 4 + j];
			}
			rot_pix[i] += center;
		}

		tx = rot_pix[0];
		ty = rot_pix[1];
		tz = rot_pix[2];

		x = tx;
		y = ty;
		z = tz;

		if (x < 1 || x > size - 2 || y < 1 || y > size - 2 || z < 1 || z > size - 2)
			continue;

		fx = tx - x;
		fy = ty - y;
		fz = tz - z;
		cx = 1. - fx;
		cy = 1. - fy;
		cz = 1. - fz;
		if (0 != slice[t])
		{
			slice[t] /= pixels[t * 4 + 3];
			w = slice[t];

			f = cx * cy * cz;
			weight[x * size * size + y * size + z] += f;
			volume_1d[x * size * size + y * size + z] += f * w;

			f = cx * cy * fz;
			weight[x * size * size + y * size + ((z + 1) % size)] += f;
			volume_1d[x * size * size + y * size + ((z + 1) % size)] += f * w;

			f = cx * fy * cz;
			weight[x * size * size + ((y + 1) % size) * size + z] += f;
			volume_1d[x * size * size + ((y + 1) % size) * size + z] += f * w;

			f = cx * fy * fz;
			weight[x * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f;
			volume_1d[x * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f * w;

			f = fx * cy * cz;
			weight[((x + 1) % size) * size * size + y * size + z] += f;
			volume_1d[((x + 1) % size) * size * size + y * size + z] += f * w;

			f = fx * cy * fz;
			weight[((x + 1) % size) * size * size + y * size + ((z + 1) % size)] += f;
			volume_1d[((x + 1) % size) * size * size + y * size + ((z + 1) % size)] += f * w;

			f = fx * fy * cz;
			weight[((x + 1) % size) * size * size + ((y + 1) % size) * size + z] += f;
			volume_1d[((x + 1) % size) * size * size + ((y + 1) % size) * size + z] += f * w;

			f = fx * fy * fz;
			weight[((x + 1) % size) * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f;
			volume_1d[((x + 1) % size) * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f * w;
		}

		else if (0 == slice[t])
		{
			f = cx * cy * cz;
			weight[x * size * size + y * size + z] += f;

			f = cx * cy * fz;
			weight[x * size * size + y * size + ((z + 1) % size)] += f;

			f = cx * fy * cz;
			weight[x * size * size + ((y + 1) % size) * size + z] += f;

			f = cx * fy * fz;
			weight[x * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f;

			f = fx * cy * cz;
			weight[((x + 1) % size) * size * size + y * size + z] += f;

			f = fx * cy * fz;
			weight[((x + 1) % size) * size * size + y * size + ((z + 1) % size)] += f;

			f = fx * fy * cz;
			weight[((x + 1) % size) * size * size + ((y + 1) % size) * size + z] += f;

			f = fx * fy * fz;
			weight[((x + 1) % size) * size * size + ((y + 1) % size) * size + ((z + 1) % size)] += f;
		}
	}
	free(slice);
}


void Data2Slice(double pattern_cart[size][size], double *slice, int size)
{
	// 根据二维衍射图数据，将二维衍射强度data变成一个一维数组slice
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			long t = i * size + j;
			slice[t] = pattern_cart[i][j];
		}
	}
}

void Slice2Data(double pattern_cart[size][size], double *slice, int size)
{
	// 根据二维衍射图数据，将二维切片一维数组slice变成二维切片reference
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			long t = i * size + j;
			pattern_cart[i][j] = slice[t];
		}
	}
}

void Hdf5Write3D(hid_t file, int dim[3], char *dataset_name, double ***data)
{
	hsize_t dimsf[3];
	hid_t dataspace, dataset;
	herr_t status;
	dimsf[0] = dim[0];
	dimsf[1] = dim[1];
	dimsf[2] = dim[2];

	dataspace = H5Screate_simple(3, dimsf, NULL);
	dataset = H5Dcreate(file, dataset_name, H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Dclose(dataset);
	H5Sclose(dataspace);
}

void Hdf5Write2D(hid_t file, int dim[2], char *dataset_name, double **data)
{
	hsize_t dimsf[2];
	hid_t dataspace, dataset;
	herr_t status;
	dimsf[0] = dim[0];
	dimsf[1] = dim[1];

	dataspace = H5Screate_simple(2, dimsf, NULL);
	dataset = H5Dcreate(file, dataset_name, H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Dclose(dataset);
	H5Sclose(dataspace);
}

void Hdf5Write1D(hid_t file, int dim[1], char *dataset_name, double *data)
{
	hsize_t dimsf[1];
	hid_t dataspace, dataset;
	herr_t status;
	dimsf[0] = dim[0];

	dataspace = H5Screate_simple(1, dimsf, NULL);
	dataset = H5Dcreate(file, dataset_name, H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	H5Dclose(dataset);
	H5Sclose(dataspace);
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
