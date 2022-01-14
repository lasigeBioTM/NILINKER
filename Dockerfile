FROM tensorflow/tensorflow:2.5.1-gpu  

RUN apt-get update && apt-get install -y git && apt-get autoclean -y
RUN apt-get update && apt-get install -y wget && apt-get autoclean -y
RUN apt-get update && apt-get install -y nano && apt-get autoclean -y
RUN apt-get update && apt-get install -y vim && apt-get autoclean -y

# Java
RUN apt-get update && apt-get install -y default-jdk && apt-get autoclean -y

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt

#scikit-learn
RUN pip3 install -U scikit-learn==0.24.2

# Download model for spacy
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz

# Install R
RUN apt-get install r-base
RUN apt-get install r-cran-car

RUN apt-install wget