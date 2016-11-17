FROM tensorflow/tensorflow:0.10.0rc0

RUN pip install dask[complete]
RUN pip install seaborn bokeh
RUN pip install scipy --upgrade
RUN pip install keras
RUN pip install mlpy

