#!/bin/bash

#文件路径参数
pattern_density_file="/density_1a9x.txt"
output_path="/1a9x_pattern_1"

#模拟衍射图参数
n_pattern=10000  
lambda=1e-10 
photon_num=1e+12  
beamsize=0.1e-6 
size=512 
z_det=1.2 
pix_len=300e-6 
beam_mask=0 
noise=1

#蛋白质电子密度参数
n_fast=154
n_medium=240
n_slow=278

#模拟衍射图进程数量
np=8 

#----------下面无需修改----------
mkdir ${output_path}/pattern_512

#画出前三张衍射图
while true;
do
    if [ -e "${output_path}/pattern_512/pattern_2.h5" ];then
	/home/jiao/anaconda3/envs/orientation/bin/python \
./plot_pattern.py \
--i=${output_path}/pattern_512 \
--size=${size} \
--pix_len=${pix_len} \
--z_det=${z_det} \
--lambda=${lambda} \
--plot_all=0 \
--plot_first_3=1
	break
    else
	sleep 5
    fi
done &

#画出所有衍射图叠加结果
while true;
do
    if [ -e "${output_path}/pattern_512/angle.h5" ];then
	/home/jiao/anaconda3/envs/orientation/bin/python \
./plot_pattern.py \
--i=${output_path}/pattern_512 \
--size=${size} \
--pix_len=${pix_len} \
--z_det=${z_det} \
--lambda=${lambda} \
--plot_all=1 \
--plot_first_3=0 \
--n_pattern=${n_pattern}
	break
    else
	sleep 60
    fi
done &

#生成衍射图
mpirun \
    -np ${np} \
    ./make_pattern_mpi \
    --density_file=${pattern_density_file} \
    --output_path=${output_path}/pattern_512 \
    --n_pattern=${n_pattern}  \
    --lambda=${lambda} \
    --photon_num=${photon_num}  \
    --beamsize=${beamsize} \
    --size=${size} \
    --z_det=${z_det} \
    --pix_len=${pix_len} \
    --beam_mask=${beam_mask} \
    --na=${n_fast} \
    --nb=${n_medium} \
    --nc=${n_slow} \
    --noise=${noise} 



