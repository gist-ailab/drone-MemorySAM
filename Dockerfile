FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 1. 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    wget git curl vim unzip build-essential \


# 2. Miniconda 설치
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh
ENV PATH $CONDA_DIR/bin:$PATH

# 3. Conda 환경 생성
RUN conda create -n MMSS_SAM python=3.10 -y
ENV CONDA_DEFAULT_ENV MMSS_SAM
ENV PATH $CONDA_DIR/envs/MMSS_SAM/bin:$PATH

# 4. PyTorch 및 필수 라이브러리 설치
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
RUN pip install numpy setuptools

# 5. 작업 경로 설정 (비워둠)
WORKDIR /workspace

CMD ["/bin/bash"]