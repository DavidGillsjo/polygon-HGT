#!/bin/bash
#Usage: [ENV_OPTS] ./run_local [CMD] [ARGS]
USE_IMAGE_USERS=1 DUSER=rapids USE_NVIDIA=1 IMAGE=${IMAGE-cuspatial} ./../libs/dockers/common/run.sh "$@"
