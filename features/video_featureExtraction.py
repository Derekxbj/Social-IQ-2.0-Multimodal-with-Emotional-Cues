import os


# create folder if not exists
if not os.path.exists("video_features"):
    os.makedirs("video_features")
        
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, VideoMAEModel
model_ckpt = "MCG-NJU/videomae-base"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", output_hidden_states=True)

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

# num_frames_to_sample = model.config.num_frames
num_frames_to_sample = 16
print(f"Number of frames to sample: {num_frames_to_sample}")
sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps
print(f"Clip duration: {clip_duration} seconds.")

train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(resize_to),
                    RandomHorizontalFlip(p=0.5),
                ]
            ),
        ),
    ]
)


import torch
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
# iterate over all videos
for video in os.listdir("../../siq2/video"):
    print("video: ", video)
    vid_name = video.split(".")[0]
    
    video = EncodedVideo.from_path('../../siq2/video/{}.mp4'.format(vid_name))
    
    # get the lenght of the video
    video_length = video.duration
    
    # split the video length into 21 evenly spaced timestamps
    timestamps = np.linspace(0, video_length, 21)
    
    video_tensors = []
    for i in range(len(timestamps)):
        if i==0:
            continue
        else:
            start_sec = timestamps[i-1]
            end_sec = timestamps[i]
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            video_features = train_transform(video_data)["video"]
            video_tensor = video_features.permute(1, 0, 2, 3)
            video_tensors.append(video_tensor)
    
    video_tensors = torch.stack(video_tensors)
    
    # save features to disk
    np.save("./video_features/{}.npy".format(vid_name), video_tensors.numpy())

