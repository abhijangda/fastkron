#!/usr/bin/env bash

ENV_NAME="krongpu"
export PRJ=${HOME}/"krongpu"
ENV_BIN_PATH=${HOME}"/anaconda3/envs/"${ENV_NAME}"/bin"

export py="${ENV_BIN_PATH}/python"
export ipy="${ENV_BIN_PATH}/ipython"
export jlab="${ENV_BIN_PATH}/jupyter-lab"
export pip="${ENV_BIN_PATH}/pip"
export nose="${ENV_BIN_PATH}/nosetests"
export wandb="${ENV_BIN_PATH}/wandb"
export gdown="${ENV_BIN_PATH}/gdown"
export PYTHONPATH=`pwd`:$PYTHONPATH
