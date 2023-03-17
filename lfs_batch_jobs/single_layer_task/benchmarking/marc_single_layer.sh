#!/bin/bash

# cmmd: bash marc_single_layer.sh 0 24 
# TODO - update path to experiment_name, logging, error logging, interp ranges, extrap ranges, id
# Setup to run only ONE job at a time (i.e. 1 seed for 1 range)

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
#    experiment_name='benchmark/sltr-in2/mul/MulMCFC'
#    mkdir -p /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/errors
#    python3 -u /home/bm4g15/nalm-benchmark/experiments/single_layer_benchmark.py \
#    --id 32 --operation mul --layer-type MulMCFC \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err # do not send to bkg. A single run takes up most of the cpu resources. 

#    experiment_name='benchmark/sltr-in2/add/MCFC'
#    mkdir -p /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/errors
#    python3 -u /home/bm4g15/nalm-benchmark/experiments/single_layer_benchmark.py \
#    --id 33 --operation add --layer-type MCFC \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err # do not send to bkg. A single run takes up most of the cpu resources.

    experiment_name='benchmark/sltr-in2/mul/MulMCFCSignINALU'
    mkdir -p /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/errors
    python3 -u /home/bm4g15/nalm-benchmark/experiments/single_layer_benchmark.py \
    --id 34 --operation mul --layer-type MulMCFCSignINALU \
    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
    > /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
    2> /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err # do not send to bkg. A single run takes up most of the cpu resources.

#    experiment_name='benchmark/sltr-in2/mul/MulMCFCSignRealNPU'
#    mkdir -p /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/errors
#    python3 -u /home/bm4g15/nalm-benchmark/experiments/single_layer_benchmark.py \
#    --id 35 --operation mul --layer-type MulMCFCSignRealNPU \
#    --regualizer-scaling-start 20000 --regualizer-scaling-end 35000 \
#    --reg-scale-type madsen --regualizer-gate 1 \
#    --interpolation-range ${interpolation_ranges[i]} --extrapolation-range ${extrapolation_ranges[i]} \
#    --seed ${seed} --max-iterations 50000 ${verbose_flag} --log-interval ${log_interval} \
#    --name-prefix ${experiment_name} --remove-existing-data --no-cuda ${no_save_flag} \
#    > /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/${interpolation_ranges[i]}-${seed}.out \
#    2> /data/bm4g15/nalu-stable-exp/logs/${experiment_name}/errors/${interpolation_ranges[i]}-${seed}.err # do not send to bkg. A single run takes up most of the cpu resources.


  done
  wait

done
wait
date
echo "Script finished."

