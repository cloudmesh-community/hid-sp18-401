FROM gliderlabs/alpine:3.3

RUN apk add --update \
    python \
    python-dev \
    py-pip \
    build-base \
    git \
    openjdk7-jre \

 && pip install --upgrade pip \  
 && pip install -U pandas numpy scikit-learn virtualenv \
 && rm -rf /var/cache/apk/*

    

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
