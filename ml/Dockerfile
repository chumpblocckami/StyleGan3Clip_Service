FROM nvidia/cuda:10.2-base

WORKDIR app

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install git
RUN apt-get install ninja-build
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip
RUN python3 -m pip install -U --force-reinstall pip

RUN git clone https://github.com/NVlabs/stylegan3
RUN git clone https://github.com/openai/CLIP

RUN pip3 install --upgrade torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -e ./CLIP
RUN pip install einops 
RUN pip install ninja

RUN pip3 install flask
RUN pip3 install wget
RUN pip3 install requests
RUN pip3 install scipy

RUN rm -rf /tmp/torch_extensions
COPY . .

RUN mkdir ./data/

EXPOSE 8000

ENTRYPOINT ["python3","app.py"]
