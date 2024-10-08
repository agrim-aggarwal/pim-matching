FROM 208876916689.dkr.ecr.us-west-2.amazonaws.com/infra/python:3.7
COPY requirements.txt requirements.txt

# Install AWS CLI dependencies
RUN apt-get -y update && \
    apt-get -y install \
        unzip \
        groff \
        curl \
        wget \
        pkg-config \
        libgl1 \
        && \
    apt-get clean all
    
# Download and install AWS CLI using pip
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install    
    
# Verify AWS CLI installation
RUN aws --version

# Set the desired AWS region
RUN aws configure set region us-west-2


COPY utils utils
COPY entrypoints entrypoints
COPY setup.py setup.py
COPY cli.py cli.py
COPY constants constants
# RUN export CFLAGS="-std=c++11"
# RUN sudo apt-get groupinstall 'Development Tools' -y
# RUN conda install libgcc -y
# RUN sudo apt-get -y install libstdc++6
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
# RUN conda config --set show_channel_urls yes
# RUN conda install paddlepaddle==2.5.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -y
# RUN pip install paddlepaddle paddleocr

RUN pip install -r requirements.txt
# RUN pip install --upgrade sagemaker
RUN pip install -e . --verbose

# RUN python setup.py install
RUN pip list
