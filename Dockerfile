FROM tensorflow/tensorflow:1.4.0-py3

MAINTAINER Olav Nymoen (olav@olavnymoen.com)

RUN mkdir /infer
ADD requirements.txt /infer/
WORKDIR /infer
RUN pip install -r requirements.txt

# Optional generate groc proto files
# RUN pip3 install grpcio-tools && cd proto && \
#     python -m grpc_tools.protoc -I. --python_out=../infer --python_grpc_out=../infer service.proto

ADD . /infer

ENTRYPOINT ["python", "src/infer.py"]

EXPOSE 8080
