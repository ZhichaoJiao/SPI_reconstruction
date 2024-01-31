
volume_file="/merge_volume_512.h5"
output_2="/recover_density"

for (( i=1; i<11; i=i+1 )); do
 ./phasing_omp \
    --volume_file=${volume_file} \
    --output_path="${output_2}/test_${i}" \
    --phase_file="/volume_6zfp_af2.h5" \
    --input_phase=0 \
    --size=512 \
    --n_bin=2 \
    --beam_stop=0 \
    --support_size=60 \
    --n_hio=0 \
    --n_er=1100 \
    --beta=0.9 
done










