FROM rapidsai/rapidsai:cuda11.5-runtime-ubuntu20.04-py3.8

# Arguments
ARG user=pytorch
ARG uid=1000
ARG gid=1000
ARG gpu_arch=sm_70

ENV DEBIAN_FRONTEND noninteractive

# Install some dependencies
RUN source activate rapids && \
		apt-get update && apt-get install -y  \
    sudo\
		build-essential\
		gdb\
		python3-dbg\
		vim
	# apt-get clean && \
	# apt-get autoremove && \
	# rm -rf /var/lib/apt/lists/*


#Install repo dependencies
# RUN pip3 install --upgrade pip &&\
# 		pip3 install --no-cache-dir\

# RUN source activate rapids && \
# 		conda config --set channel_priority true &&\
# 		conda install --yes cudatoolkit=11.3

# RUN source activate rapids && \
# 		CONDA_OVERRIDE_CUDA="11.3" conda install --yes --override-channels \
# 		-c pytorch -c main\
# 		pytorch=1.12.1  torchvision

RUN source activate rapids && \
		conda install -c fastai \
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

RUN source activate rapids && \
		pip3 install --no-cache-dir torch torchvision  --extra-index-url https://download.pytorch.org/whl/cu113
RUN source activate rapids && \
		pip3 install --no-cache-dir \
	  torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

RUN source activate rapids &&\
 		conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/conda/envs/rapids/lib

# RUN source activate rapids && \
		# conda install  -y --override-channels -c conda-forge -c "nvidia/label/cuda-11.5.0" cuda-nvcc gxx_linux-64
		# conda install  -y --override-channels -c conda-forge nvcc_linux-64

# Setup user
RUN source activate rapids && \
		export uid="${uid}" gid="${gid}" && \
    groupadd -g "${gid}" "${user}" && \
    useradd -m -u "${uid}" -g "${user}" -s /bin/bash "${user}" && \
    passwd -d "${user}" && \
    usermod -aG sudo "${user}" &&\
		echo "sudo ALL=(ALL:ALL) ALL" >> /etc/sudoers

# Switch to user
USER "${uid}"

VOLUME /data
VOLUME /host_home

# Default powerline10k theme, no plugins installed
RUN source activate rapids && sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)"

WORKDIR /host_home
CMD zsh
