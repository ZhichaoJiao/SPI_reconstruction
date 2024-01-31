#!/bin/bash

#文件路径参数
volume_density_file="/density_8it1_af2.txt"
output_path="/8it1"
output_volume_file="volume_8it1_af2.h5"

#模拟衍射图参数
lambda=1e-10 
z_det=1 
pix_len=300e-6 
size=512

#电子密度图参数
n_fast=210
n_medium=250
n_slow=224

#----------下面无需修改----------

./make_volume \
    --density_file=${volume_density_file} \
    --file_out="${output_path}/${output_volume_file}" \
    --z_det=${z_det} \
    --lambda=${lambda} \
    --pix_len=${pix_len} \
    --size=${size} \
    --na=${n_fast} \
    --nb=${n_medium} \
    --nc=${n_slow} 

