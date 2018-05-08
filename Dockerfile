FROM tensorflow/tensorflow:1.8.0-py3

MAINTAINER Olav Nymoen (olav@olavnymoen.com)

RUN pip install tfweb

ENTRYPOINT ["tfweb"]

EXPOSE 8080
