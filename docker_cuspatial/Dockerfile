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
RUN conda install  -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

USER root
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir \
	  torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod +xr /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]

# Back to default user
USER rapids

VOLUME /data
VOLUME /host_home

# Default powerline10k theme, no plugins installed
# WORKDIR "/tmp"
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"

WORKDIR /host_home
CMD bash
