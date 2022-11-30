FROM nvidia/cuda:11.3.0-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y build-essential curl git wget gosu sudo
RUN apt-get install -y build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python3.8
RUN apt-get install -y python3.8-dev
RUN apt-get install -y libjpeg-dev
RUN python3.8 -m pip install cython
RUN python3.8 -m pip install numpy
RUN python3.8 -m pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip install torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip install torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip install pyyaml
RUN python3.8 -m pip install pandas
RUN python3.8 -m pip install IPython
RUN python3.8 -m pip install packaging
RUN python3.8 -m pip install markupsafe==1.1.1
RUN python3.8 -m pip install jupyter
RUN python3.8 -m pip install matplotlib
RUN python3.8 -m pip install scipy==1.5.0
RUN python3.8 -m pip install -U pip
RUN python3.8 -m pip install learn2learn==0.1.6
RUN python3.8 -m pip install requests
RUN python3.8 -m pip install torchmeta==1.8.0
RUN python3.8 -m pip install h5py

ARG USER_ID
ARG GROUP_ID
RUN echo $GROUP_ID
RUN addgroup --gid $GROUP_ID docker
RUN adduser --disabled-password --gecos ''  --uid $USER_ID --gid $GROUP_ID docker

USER docker
