FROM floydhub/dl-docker:cpu

RUN pip install dask[complete]
RUN pip install seaborn
RUN pip install scipy --upgrade
