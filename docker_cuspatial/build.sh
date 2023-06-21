#!/bin/bash
USE_NVIDIA=1 IMAGE=${IMAGE-cuspatial} ./../libs/dockers/common/build.sh "$@"
