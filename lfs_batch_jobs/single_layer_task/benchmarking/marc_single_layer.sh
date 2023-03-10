#!/bin/bash

# cmmd: bash marc-all-ranges.sh 0 0 ; bash marc-all-ranges.sh 1 1
# TODO - update path to experiment_name, logging, error logging, interp ranges, extrap ranges, id

export LSB_JOB_REPORT_MAIL=N


verbose_flag=''
no_save_flag=''
log_interval='1000'

interpolation_ranges=( '[-20,-10]' '[-2,-1]' '[-1.2,-1.1]' '[-0.2,-0.1]' '[-2,2]'          '[0.1,0.2]' '[1,2]' '[1.1,1.2]' '[10,20]' )
extrapolation_ranges=( '[-40,-20]' '[-6,-2]' '[-6.1,-1.2]' '[-2,-0.2]'  '[[-6,-2],[2,6]]' '[0.2,2]'   '[2,6]' '[1.2,6]'  '[20,40]' )
#interpolation_ranges=( '[1,2]' )
#extrapolation_ranges=( '[2,6]' )


for ((i=0;i<${#interpolation_ranges[@]};++i))
  do
  for seed in $(eval echo {$1..$2})
    do

    export TENSORBOARD_DIR=/data/bm4g15/nalu-stable-exp/tensorboard
    export SAVE_DIR=/data/bm4g15/nalu-stable-exp/saves
    export PYTHONPATH=./

    # TODO - uncomment the relevant experiment and run.
    experiment_name='benchmark/sltr-in2/MulMCFC'
    mkdir -p /data/nalms/logs/${experiment_name}/errors
    python3 -u experiments/single_layer_benchmark.py \
    --id 0 --operation mul --layer-type MulMCFC \
    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &

#    experiment_name='benchmark/sltr-in2/MCFC'
#    mkdir -p /data/nalms/logs/${experiment_name}/errors
#    python3 -u experiments/single_layer_benchmark.py \
#    --id 1 --operation add --layer-type MCFC \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/nalms/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/nalms/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err &
  done
  wait

done
wait
date
echo "Script finished."
