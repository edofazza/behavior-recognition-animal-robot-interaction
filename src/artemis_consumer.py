import os

os.environ['HF_HOME'] = './.cache'
import torch
import redis
from timesformer import TimeSformer
from PIL import Image
from torchvision.transforms import Compose
from .transforms_ss import *


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


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


def get_transform():
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    input_size = 224
    scale_size = 256
    unique = Compose([GroupScale(scale_size),
                      GroupCenterCrop(input_size)])
    common = Compose([Stack(roll=False),
                      ToTorchFormatTensor(div=True),
                      GroupNormalize(input_mean, input_std)])
    transforms = Compose([unique, common])
    return transforms

if __name__ == '__main__':
    # Variables
    stream_name = 'detections'
    group_name = 'artemis_group'
    consumer_name = 'artemis_consumer'
    model_path = 'timesformer_dynamic.pth'
    # Prepare output directory
    save_dir = './action_inference'
    os.makedirs(save_dir, exist_ok=True)
    # Connect to Redis container
    r = connect_to_redis(stream_name, group_name)
    # Load model
    device = get_device()
    model = TimeSformer(16).to(device)
    model.load_state_dict(torch.load('timesformer_dynamic.pth'))
    model.eval()
    # Transform
    transform = get_transform()
    # Frame storing
    sequence_buffer = []
    frames_idx = []

    while True:
        entries = r.xreadgroup(group_name, consumer_name, {stream_name: '>'}, count=1, block=0)
        if entries:
            for stream, messages in entries:
                for msg_id, msg in messages:
                    frame_path = msg[b'frame_path'].decode()
                    object_present = msg.get(b'object_present', b'0').decode() == "1"

                    if object_present:
                        try:
                            sequence_buffer.extend([Image.open(frame_path).convert('RGB')])
                            frames_idx.append(frame_path.split("/")[-1][:-4].split('_')[1])
                        except Exception as e:
                            print(f'ERROR: Could not read image {frame_path}')

                        print(f"ARTEMIS: Added {frame_path} (buffer size: {len(sequence_buffer)})")

                        if len(sequence_buffer) == 16:
                            print("ARTEMIS: 16 consecutive frames detected. Running action recognition on:")
                            # sequence buffer shape: (16, height, width, 3)
                            model_input = transform(sequence_buffer)
                            model_input = model_input.view((16, -1) + model_input.size()[-2:])
                            model_input = model_input.unsqueeze(0).to(device)
                            output = model(model_input)

                            # Save probabilities in numpy
                            sequence_name = '_'.join(frames_idx)
                            output = output.cpu().detach().numpy()
                            np.save(os.path.join(save_dir, sequence_name + '.npy'), output)

                            # Remove oldest frame
                            sequence_buffer.pop(0)
                            frames_idx.pop(0)
                    else:
                        if sequence_buffer:
                            print(f"ARTEMIS: Object lost at {frame_path}. Resetting sequence buffer.")
                        sequence_buffer = []
                        frames_idx = []

                    r.xack(stream_name, group_name, msg_id)
