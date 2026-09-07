FROM docker.io/miguelaeh/pipeless 

RUN pip3 install --no-cache-dir --break-system-packages numpy opencv-python onnxruntime

WORKDIR /app
COPY . /app/

ENTRYPOINT ["pipeless", "start", "-p"]
CMD ["/app/pred-pipe"]