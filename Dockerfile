FROM node:latest
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/app/
COPY . .

RUN apt update
RUN git clone https://github.com/p-balu/picSketecher-frontend


WORKDIR /home/app/picSketecher-frontend/
RUN npm install

EXPOSE 8000
EXPOSE 8080

WORKDIR /home/app/
RUN git clone https://github.com/chaitanyaambarukhana/picsketcher.git
WORKDIR /home/app/picsketcher/
RUN git branch -f nikhil origin/nikhil
RUN git checkout nikhil

RUN apt install -y python3-pip
RUN apt install -y ffmpeg libsm6 libxext6
RUN pip3 install -r requirements.txt
RUN chmod +x manage.py

WORKDIR /home/app/picsketcher/registration/

WORKDIR /home/app/

RUN chmod +x run.sh

CMD ["sh","run.sh"]