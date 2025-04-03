import pyrealsense2 as rs
import numpy as np
import cv2
import redis
import os


def initialize_realsense_pipeline():
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Get device and color sensor
    device = profile.get_device()
    color_sensor = device.first_color_sensor()

    # Enable Auto-Exposure Priority
    color_sensor.set_option(rs.option.auto_exposure_priority, 1)

    # Enable HDR mode (if supported)
    if color_sensor.supports(rs.option.hdr_enabled):
        color_sensor.set_option(rs.option.hdr_enabled, 1)
        print("HDR Mode Enabled")

    return pipeline


def consume_frames(pipeline, n_frames):
    for _ in range(n_frames):
        pipeline.wait_for_frames()
    return


if __name__ == '__main__':
    # Initialize the RealSense pipeline
    pipeline = initialize_realsense_pipeline()
    # Connect to Redis
    r = redis.Redis(host='localhost', port=6379, db=0)

    frame_counter = 0
    save_dir = './frames'
    os.makedirs(save_dir, exist_ok=True)
    skip_frames = int(30 * 0.375)

    try:
        # Wait for frames (give some time for exposure adjustments)
        consume_frames(pipeline, 60)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                raise RuntimeError("Could not acquire color frame.")

            # Convert to NumPy array and save
            color_image = np.asanyarray(color_frame.get_data())
            frame_path = os.path.join(save_dir, f'frame_{frame_counter}.jpg')
            cv2.imwrite(frame_path, color_image)

            # Add the frame to the Redis stream
            r.xadd('frames', {'frame_path': frame_path})
            print(f"REALSENSE CAMERA: Published {frame_path}")
            frame_counter += 1
            consume_frames(pipeline, skip_frames)

    finally:
        # Stop the pipeline
        pipeline.stop()
