#!/bin/bash
USE_NVIDIA=1 IMAGE=${IMAGE-cuspatial_old} ./../libs/dockers/common/build.sh "$@"
