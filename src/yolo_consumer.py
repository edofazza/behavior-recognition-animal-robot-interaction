import cv2
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from typing import Set, List, Optional
import redis
import os
import pandas as pd


def connect_to_redis(stream_name, group_name):
    r = redis.Redis(host='localhost', port=6379, db=0)

    try:
        r.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except redis.exceptions.ResponseError as e:
        if 'BUSYGROUP' in str(e):
            print('YOLO: Consumer group already exists')
        else:
            raise e
    return r


def check_presence(boxes: List[Boxes], class_ids: Set[int], confidence: float) -> List[Boxes]:
    valid_boxes = []
    for box in boxes:
        if int(box.cls.item()) not in class_ids or box.conf < confidence:
            continue
        valid_boxes.append(box)
    return valid_boxes


def convert_valid_boxes(valid_boxes: List[Boxes]) -> Optional[dict]:
    new_boxes = dict()

    if len(valid_boxes) == 0:
        return None
    elif len(valid_boxes) == 1:
        new_boxes['cls'] = valid_boxes[0].cls
        new_boxes['conf'] = valid_boxes[0].conf
        new_boxes['data'] = valid_boxes[0].data
        new_boxes['xyxy'] = valid_boxes[0].xyxy

    if len(valid_boxes) > 1:
        for box in valid_boxes[1:]:
            new_boxes['cls'] = torch.cat((valid_boxes[0].cls, box.cls))
            new_boxes['conf'] = torch.cat((valid_boxes[0].conf, box.conf))
            new_boxes['data'] = torch.cat((valid_boxes[0].data, box.data))
            new_boxes['xyxy'] = torch.cat((valid_boxes[0].xyxy, box.xyxy))

    return new_boxes


if __name__ == '__main__':
    # Variables
    yolo_path = 'yolo11s.pt'
    input_stream = 'frames'
    output_stream = 'detections'
    group_name = 'yolo_group'
    consumer_name = 'yolo_consumer'
    # Prepare output directory
    save_dir = './yolo_inference'
    os.makedirs(save_dir, exist_ok=True)
    # Connect to Redis container
    r = connect_to_redis(input_stream, group_name)
    # Load model
    model = YOLO(yolo_path)
    model.to('cuda')

    # Consume and process
    while True:
        entries = r.xreadgroup(group_name, consumer_name, {input_stream: '>'}, count=1, block=0)
        if entries:
            for stream, messages in entries:
                for msg_id, msg in messages:
                    # Get path to image
                    frame_path = msg[b'frame_path'].decode()
                    # Load image
                    img = cv2.imread(frame_path)
                    # Process with YOLO, get [0] since only one image is processed
                    result = model(img)[0]
                    # Get meaningful bounding boxes
                    valid_boxes_dict = convert_valid_boxes(check_presence(result.boxes, {18, 19}, 0.8))
                    # Publish the detection results to the 'detections' stream read by ARTEMIS
                    message = {
                        'frame_path': frame_path,
                        'object_present': '1' if valid_boxes_dict else '0'
                    }
                    r.xadd(output_stream, message)
                    print(f"YOLO: Processed {frame_path} and published detection: {message}")
                    # Acknowledge the message from the 'frames' stream
                    r.xack(input_stream, group_name, msg_id)
                    # Save YOLO result information
                    if valid_boxes_dict:
                        boxes = valid_boxes_dict['xyxy'].tolist()
                        classes = valid_boxes_dict['cls'].tolist()
                        names = result.names
                        confidences = valid_boxes_dict['conf'].tolist()
                        for box, cls, conf in zip(boxes, classes, confidences):
                            output_dict = {
                                'class': names[int(cls)],
                                'conf': float(conf),
                                'bbox': box
                            }
                            pd.DataFrame([output_dict]).to_csv(
                                os.path.join(save_dir, f'{frame_path.split("/")[-1][:-4]}.csv'), index=False)




