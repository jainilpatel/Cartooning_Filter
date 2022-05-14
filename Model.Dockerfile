FROM tensorflow/tensorflow:1.15.5-gpu
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt -y upgrade
RUN apt install -y git

WORKDIR /home/
RUN git clone https://github.com/TachibanaYoshino/AnimeGANv2

WORKDIR /home/AnimeGANv2
RUN mkdir images

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install opencv-python
RUN pip3 install tqdm
RUN pip3 install flask
RUN pip3 install Pillow
RUN pip3 install flask-cors

COPY custom_test.py /home/AnimeGANv2/custom_test.py
EXPOSE 5000

CMD ["python3","custom_test.py"]