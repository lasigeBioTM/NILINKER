FROM tensorflow/tensorflow:2.5.1-gpu  

RUN apt-get update && apt-get install -y git && apt-get autoclean -y
RUN apt-get update && apt-get install -y wget && apt-get autoclean -y
RUN apt-get update && apt-get install -y nano && apt-get autoclean -y
RUN apt-get update && apt-get install -y vim && apt-get autoclean -y
RUN apt-get update && apt-get install -y unrar && apt-get autoclean -y

# Java
RUN apt-get update && apt-get install -y default-jdk && apt-get autoclean -y

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt

#scikit-learn
RUN pip3 install -U scikit-learn==0.24.2

# Install R
#RUN apt-get install r-base
#RUN apt-get install r-cran-car