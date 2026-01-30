FROM docker.io/miguelaeh/pipeless 

RUN pip3 install --no-cache-dir --break-system-packages numpy opencv-python

COPY . /app/
WORKDIR /app

ENTRYPOINT [ "pipeless", "start", "-p", "/app/pred-pipe" ]