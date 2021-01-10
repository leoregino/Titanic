FROM manifoldai/orbyter-ml-dev:3.1
ADD requirements.txt /build/requirements.txt
RUN apt update && apt install fonts-humor-sans -y \
    && fc-cache -f
WORKDIR /build/
RUN pip install -r requirements.txt
WORKDIR /mnt/
