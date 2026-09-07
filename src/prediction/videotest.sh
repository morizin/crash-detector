mediamtx mediamtx.yml &
MEDIAMTX=$!

sleep 2

ffmpeg -re -i output.avi \
  -c:v copy \
  -f rtsp rtsp://localhost:8554/playback & 

PID=$!
trap "kill -9 $PID" INT TERM
trap "kill -9 $MEDIAMTX" INT TERM EXIT

# curl -X POST http://localhost:3030/streams \
#   -H "Content-Type: application/json" \
#   -d '{
#     "input_uri": "rtsp://host.docker.internal:8554/playback",
#     "output_uri": "rtmp://host.docker.internal:1935/out",
#     "frame_path": ["my-stage"]
#   }'

wait $PID