# Real-Time Behavior Recognition for Animal-Robot Interaction

This is a step-by-step guide to use the components described in the paper on the Go2 Unitree robot.

## Redis

Redis container for communication can be started using:

```bash
docker run -d --rm --name redis --network host redis:latest
```

## YOLO

Dockerfile:

```bash
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

RUN apt-get update
RUN pip install "opencv-python-headless<4.3"
RUN pip install ultralytics
RUN pip install transformers==4.46.3
RUN pip install redis
RUN pip uninstall -y opencv-python-headless opencv-python
RUN pip install "opencv-python-headless<4.3"

COPY . .
CMD ["python3", "yolo_consumer.py"]
```

To install image:

To run container:

```bash
cd yolo
docker build -t yolo-consumer .
```

```bash
docker run -it --runtime nvidia --name yolo --rm --network host -v $(pwd)/../frames:/app/frames -v $(pwd)/../yolo_inference:/app/yolo_inference -e REDIS_HOST=localhost -e REDIS_PORT=6379 yolo-consumer
```

## ARTEMIS

Dockerfile:

```bash
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /app

RUN apt-get update
RUN pip install "opencv-python-headless<4.3"
RUN pip install ultralytics
RUN pip install transformers==4.46.3
RUN pip install redis
RUN pip uninstall -y opencv-python-headless opencv-python
RUN pip install "opencv-python-headless<4.3"

COPY . .
CMD ["python3", "artemis_consumer.py"]
```

To install image:

```bash
cd artemis
docker build -t artemis-consumer .
```

To run container:

```bash
docker run -it --runtime nvidia --name artemis --rm --network host -v $(pwd)/../frames:/app/frames -v $(pwd)/../action_inference:/app/action_inference -e REDIS_HOST=localhost -e REDIS_PORT=6379 artemis-consumer
```

## DualSense Camera

Since we are not using any container for the camera, we can run the code in detached mode by simply:

```bash
python3 camera_acquisition.py &
```