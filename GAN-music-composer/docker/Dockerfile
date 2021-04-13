FROM ubuntu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update

RUN apt-get -y install  python3 \
                        python3-pip \
			vim \
			libsndfile1 

COPY . /root/gan
WORKDIR /root/gan

RUN pip3 install -r requirements.txt




