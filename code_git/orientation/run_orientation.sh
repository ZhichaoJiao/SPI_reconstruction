#!/bin/bash

#initial_volume_mode三种模式:(1)import_volume;(2)random_angle;(3)real_angle.

#output_6 
/usr/lib64/mpich/bin/mpirun \
    -np 8 \
    ./orientation_no_weight \
    --size=100 \
    --pattern_path="/ssd1/Data/6ZFP/6zfp_pattern_17/pattern_resize_100/pattern_" \
    --volume_path="/ssd1/Data/6ZFP/6zfp_pattern_17/pattern_resize_100/volume_6zfp_af2.h5" \
    --output_path="/ssd1/Data/6ZFP/6zfp_pattern_17/output_4/test_2" \
    --real_angle_path="/ssd1/Data/6ZFP/6zfp_pattern_17/pattern_512/angle.h5" \
    --initial_mode=1 \
    --n_pattern=20000 \
    --lambda=1e-10 \
    --z_det=0.5 \
    --pix_len=600e-6 \
    --step=0.1 \
    --r_min=6 --r_max=40 \
    --n_gamma=256 \
    --n_loop=50 \
    --fine_search=0 \
    --plot_result=1








