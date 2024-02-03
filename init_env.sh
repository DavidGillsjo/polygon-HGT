#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# If .env exists, source that too
ENV_FILE=".env"
if [ -e ${ENV_FILE} ]
then
  source ${ENV_FILE}
fi