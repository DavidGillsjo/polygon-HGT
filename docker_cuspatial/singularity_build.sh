#!/bin/bash
# Start a docker registry
RPORT=5000
DIMAGE=${IMAGE-cuspatial}

sudo docker run -d -p "$RPORT:$RPORT" --restart=always --name registry registry:2

# Push local docker container to it
sudo docker tag "$DIMAGE" "localhost:$RPORT/$DIMAGE"
sudo docker push "localhost:$RPORT/$DIMAGE"

SINGULARITY_CACHEDIR="$HOME/.singularity_cache" SINGULARITY_NOHTTPS=1 singularity build "$DIMAGE.sif" "docker://localhost:$RPORT/$DIMAGE"
# sudo SINGULARITY_CACHEDIR="$HOME/.singularity_cache" SINGULARITY_NOHTTPS=1 singularity build "$DIMAGE.sif" Singularity
