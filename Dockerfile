FROM tensorflow/tensorflow:1.4.0-py3

MAINTAINER Olav Nymoen (olav.nymoen@schibsted.com)

RUN mkdir /infer
ADD requirements.txt /infer/
WORKDIR /infer
RUN pip install -r requirements.txt

ADD . /infer

ENTRYPOINT ["python", "src/infer.py"]

EXPOSE 8080
