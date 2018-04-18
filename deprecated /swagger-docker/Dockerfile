FROM ubuntu:16.04
MAINTAINER Goutham Arra <garra@iu.edu>

RUN apt-get update -q && apt-get install -qy \
    build-essential \
    python-dev \
    python-pip \
    git-core \
    wget \
    curl \
    default-jre \
    libssl-dev -y

# install net tools
RUN pip install --upgrade pip
RUN pip install -U numpy scipy scikit-learn pandas

    

#RUN apt-get update -q && apt-get install -qy default-jre 



#RUN pip install connexion
#RUN pip install pymongo
RUN git clone https://github.com/cloudmesh-community/hid-sp18-401.git

WORKDIR /hid-sp18-401/swagger-docker

#RUN wget http://central.maven.org/maven2/io/swagger/swagger-codegen-cli/2.3.1/swagger-codegen-cli-2.3.1.jar
EXPOSE 8080

#FROM openjdk:7 
RUN make service

CMD ["make", "start"]
