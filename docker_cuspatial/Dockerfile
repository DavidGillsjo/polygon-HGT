FROM nvcr.io/nvidia/rapidsai/base:23.08-cuda11.8-py3.10

# Arguments
ARG user=pytorch
ARG uid=1000
ARG gid=1000
ARG gpu_arch=sm_80

ENV DEBIAN_FRONTEND noninteractive

USER root

# Install some dependencies
RUN apt-get update && apt-get install -y  \
    sudo\
		build-essential\
		gdb\
		python3-dbg\
		wget \
		vim
	# apt-get clean && \
	# apt-get autoremove && \
	# rm -rf /var/lib/apt/lists/*


#Install repo dependencies
# RUN pip3 install --upgrade pip &&\
# 		pip3 install --no-cache-dir\

# RUN conda activate rapids && \
# 		conda config --set channel_priority true &&\
# 		conda install --yes cudatoolkit=11.3

# RUN conda activate rapids && \
# 		CONDA_OVERRIDE_CUDA="11.3" conda install --yes --override-channels \
# 		-c pytorch -c main\
# 		pytorch=1.12.1  torchvision
USER rapids
RUN conda install -c fastai \
		opencv-python-headless \
		tensorboard \
		future \
		Cython \
		matplotlib \
		numpy \
		scipy \
		pyyaml \
		gitpython \
		seaborn \
		pycairo\
		# Structured 3D
		descartes \
		shapely \
		# HAWP
		yacs\
		requests\
		scikit-image\
		tabulate\
		networkx\
		tqdm \
		wandb \
		h5py

USER root
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir \
	  torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
USER rapids

RUN conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/conda/envs/rapids/lib

# RUN conda activate rapids && \
# 		conda install  -y --override-channels -c conda-forge -c "nvidia/label/cuda-11.8.0" cuda-nvcc gxx_linux-64
		# conda install  -y --override-channels -c conda-forge nvcc_linux-64
# USER root
# COPY docker_cuspatial/entrypoint.sh /entrypoint.sh
# RUN chmod -R +xr /home/rapids /opt/conda/etc/profile.d/conda.sh /entrypoint.sh
# ENTRYPOINT [ "/entrypoint.sh" ]
# Setup user
# RUN export uid="${uid}" gid="${gid}" && \
#     groupadd -g "${gid}" "${user}" && \
#     useradd -m -u "${uid}" -g "${user}" -s /bin/bash "${user}" && \
#     passwd -d "${user}" && \
#     usermod -aG sudo "${user}" &&\
# 	echo "sudo ALL=(ALL:ALL) ALL" >> /etc/sudoers

# Switch to user
# USER "${uid}"
USER rapids

VOLUME /data
VOLUME /host_home

# Default powerline10k theme, no plugins installed
# WORKDIR "/tmp"
# RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"

# WORKDIR /host_home
# CMD zsh
