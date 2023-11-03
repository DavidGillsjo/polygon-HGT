#!/bin/bash
#Usage: [ENV_OPTS] ./run_local [CMD] [ARGS]
USE_NVIDIA=1 IMAGE=${IMAGE-cuspatial_old} ./../libs/dockers/common/run.sh "$@"
