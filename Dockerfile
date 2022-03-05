FROM civisanalytics/datascience-python

COPY environment.yaml /
COPY requirements.txt /

RUN apt update --fix-missing

RUN conda config --append channels conda-forge
RUN conda install -y --file requirements.txt
