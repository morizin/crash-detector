#!/bin/bash
set -e

FFMPEG=""
MEDIAMTX=""

cleanup(){
    if [ ! -z "$FFMPEG" ]; then
        kill -9 $FFMPEG 2>/dev/null
        echo "Killing FFMPeG"
    fi
    if [ ! -z "$MEDIAMTX" ]; then
        kill -9 $MEDIAMTX 2>/dev/null
        echo "Killing MediaMTX" 
    fi
    exit
}

trap cleanup INT TERM EXIT

mediamtx mediamtx.yml &
MEDIAMTX=$!

sleep 2

ffmpeg -nostdin -f avfoundation \
    -video_size 640x480 \
    -framerate 30 -i "0" \
    -c:v hevc_videotoolbox \
    -b:v 1.5M -pix_fmt yuv420p \
    -f flv rtmp://localhost:1935/live \
    2>&1 &

FFMPEG=$!

wait $FFMPEG