#!/bin/bash
#Usage: [ENV_OPTS] ./run_local [CMD] [ARGS]
PULL=1 USE_NVIDIA=1 IMAGE=${IMAGE-davidgillsjo/polygon-hgt} ./../libs/dockers/common/run.sh "$@"
