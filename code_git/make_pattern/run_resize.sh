#!bash
#先bin,再截取中心的resize大小

./resize --resize_volume=1 \
        --input_volume_path="/volume_8it1_af2.h5" \
        --output_volume_path="/volume_8it1_af2_resize_100.h5" \
        --resize_pattern=0 \
        --input_pattern_path="/pattern_" \
        --output_pattern_path="/pattern_" \
        --n_pattern=20000 \
        --size=512 \
        --bin=4 \
        --resize=100

