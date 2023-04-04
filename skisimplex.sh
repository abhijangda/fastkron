#!/usr/bin/env bash

ENV_NAME="sgkigp"
export PRJ=${HOME}/"sgkigp"
ENV_BIN_PATH=${HOME}"/anaconda3/envs/"${ENV_NAME}"/bin"
py="${ENV_BIN_PATH}/python"

# iterating variables
# datasets, boundary_slack, grid_size_dim
#energy" #kin40k energy concrete fertility protein pendulum
datasets="servo houseelectric 3droad protein"  #protein elevators keggdirected  kin40k
boundary_slacks="0.005"
gsds="5"

# extra config
interp_type="2"
epochs="50"
method="0"

# log dir
log_dir="logs/"
#rm -fr ${log_dir}
mkdir -p ${log_dir}

for gsd in ${gsds}; do
  for bs in ${boundary_slacks}; do
    for dataset in ${datasets}; do
      log_dir_path=${log_dir} #/"dataset_"${dataset}"_gsd_"${gsd}"_bs_"${bs}
      mkdir -p ${log_dir_path}
      tune_args="--dataset ${dataset} --epochs ${epochs} --method ${method} --grid_size_dim ${gsd} --boundary_slack ${bs}  --interp_type ${interp_type} --log_dir ${PRJ}/${log_dir_path}"
      commands+=("$py -m experiments.runner ${tune_args}")
      echo ${tune_args} > "${PRJ}/${log_dir_path}_cmd.str"
    done
  done
done


# Running jobs
num_jobs=${#commands[@]}

for ((job_id=0; job_id<num_jobs; job_id++)); do
    comm="${commands[$job_id]}"
    echo ${job_id} ${num_jobs}
    echo ${comm}
    #eval ${comm}
done
