
FROM nvcr.io/nvidia/pytorch:23.11-py3
RUN mkdir /fastkron/
COPY . /fastkron/
