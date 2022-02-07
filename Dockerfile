FROM ucsdets/datahub-base-notebook:2021.1-stable

USER root
ARG DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /projectHome
RUN useradd -m bam -s /projectHome -s /bin/bash 
RUN chown bam -R /projectHome /home/jovyan/ /bin/bash 

RUN apt-get update --fix-missing &&\
        apt-get -y install \
        git \
        nano \
        python3-pip \
        libgdal-dev

RUN pip3 install jupyterlab\
        scikit-learn

RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal && export C_INCLUDE_PATH=/usr/include/gdal

COPY ./requirements.txt /projectHome/requirements.txt
# RUN conda create --name capstone --file /projectHome/requirements.txt
RUN pip install -r /projectHome/requirements.txt
RUN pip install "napari[all]"

USER bam
WORKDIR /home/projectHome/

RUN echo "source activate capstone" > ~/.bashrc
ENV PATH /opt/conda/envs/capstone/bin:$PATH