#!/bin/bash

cd /home/app/picSketecher-frontend/ && nohup npm start &

cd /home/app/picsketcher/ && python3 manage.py runserver 0.0.0.0:8000
