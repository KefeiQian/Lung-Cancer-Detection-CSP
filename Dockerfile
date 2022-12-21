FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

WORKDIR /root

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list

RUN apt update; exit 0

RUN apt install -y aptitude build-essential libglib2.0-0 curl libsm6 libxrender1 libfontconfig1 libxext6 vim

RUN aptitude install -y python-dev

RUN apt install -y gcc-5 g++-5
RUN ln -s /usr/bin/gcc-5 /usr/local/cuda/bin/gcc
RUN ln -s /usr/bin/g++-5 /usr/local/cuda/bin/g++

RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
RUN python get-pip.py

ADD requirements.txt requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple oss2 tqdm


# uncomment following to run in cloud services
# COPY data /root/csp/data/
# COPY eval_caltech /root/csp/eval_caltech/
# COPY keras_csp /root/csp/keras_csp/
# COPY scripts /root/csp/scripts
# COPY train_caltech.py /root/csp/
# COPY test_caltech.py /root/csp/
# COPY evaluate.py /root/csp