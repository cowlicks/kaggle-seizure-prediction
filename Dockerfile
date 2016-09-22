FROM floydhub/dl-docker:cpu

RUN pip install dask[complete]
RUN pip install seaborn bokeh
RUN pip install scipy --upgrade

