FROM tensorflow/tensorflow:1.15.5-jupyter
RUN pip install pip-tools
RUN apt-get update \
  && apt-get install -y git pkg-config libfftw3-dev liblapack-dev \
  && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt
