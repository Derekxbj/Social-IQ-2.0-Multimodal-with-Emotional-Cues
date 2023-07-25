from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import os

from utils import get_videos_not_found

from transformers import AutoTokenizer

class generateDataset():
    def __init__(self, train_path, val_path, video_path, tokenizer):

        matched_df_train = pd.read_json(path_or_buf= os.path.join(train_path),lines=True)
        matched_df_val = pd.read_json(path_or_buf= os.path.join(val_path),lines=True)
        
        videos_not_found_train, all_videos = get_videos_not_found(train_path, video_path)
        videos_not_found_val, all_videos = get_videos_not_found(val_path, video_path)
        
        # remove videos not found in the dataset
        for i,row in matched_df_train.iterrows():
            if row["vid_name"] in videos_not_found_train:
                # remove row
                matched_df_train.drop(i, inplace=True)
                
        for i,row in matched_df_val.iterrows():
            if row["vid_name"] in videos_not_found_val:
                # remove row
                matched_df_val.drop(i, inplace=True)
                
        self.dataset_train = Dataset.from_pandas(matched_df_train)
        self.dataset_val = Dataset.from_pandas(matched_df_val)
    
        self.tokenizer = tokenizer

    def preprocess_function(self, examples):
        # print(examples)
        
        answers_names = ["a0", "a1", "a2", "a3"]
        
        summary = []
        for vid_name in examples["vid_name"]:
            # read the text file
            with open("./summary/{}.txt".format(vid_name), "r") as f:
                summary.append(f.read())
                
        # concaeate the summary with the question and add a [SEP] 
        examples["q"] = [summary[i] + " </s> " + examples["q"][i] for i in range(len(examples["q"]))]
        
        first_sentences = [[context] * 4 for context in examples["q"]]
        second_sentences = [[examples[end][i] for end in answers_names] for i in range(len(examples["q"])) ]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = self.tokenizer(first_sentences, second_sentences, truncation=True)
        
        features = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        
        
        features["videos"] = examples["vid_name"]

        return features
    
    def returnDataset(self):
        
        
        tokenized_train = self.dataset_train.map(self.preprocess_function, batched=True)
        tokenized_val = self.dataset_val.map(self.preprocess_function, batched=True)
        
        tokenized_train = tokenized_train.rename_column("answer_idx", "label")
        tokenized_val = tokenized_val.rename_column("answer_idx", "label")
        
        tokenized_train = tokenized_train.remove_columns(['qid', 'q', 'vid_name', 'ts', 'ans_corr', 'idx_types', 'a0', 'a1', 'a2', 'a3', '__index_level_0__',])
        tokenized_val = tokenized_val.remove_columns(['qid', 'q', 'vid_name', 'ts', 'ans_corr', 'idx_types', 'a0', 'a1', 'a2', 'a3', '__index_level_0__',])
        
        
        return tokenized_train, tokenized_val
    
from transformers import VideoMAEImageProcessor
# model_ckpt = "MCG-NJU/videomae-base"
model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)

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


from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # print(features[0].keys())
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        videos = [feature.pop("videos") for feature in features]
        # print(videos)
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        
        video_tensors = []
        for vid_name in videos:
            videos_data = np.load('./video_features_2s/{}.npy'.format(vid_name))
            videos_tensor = torch.from_numpy(videos_data)
            video_tensors.append(videos_tensor)
        
        video_tensors = torch.stack(video_tensors)
        batch["video_tensors"] = video_tensors
        
        audio_tensors = []
        for vid_name in videos:
            audio_data = np.load('./audio_features_2s/{}.npy'.format(vid_name))
            audio_tensor = torch.from_numpy(audio_data)
            audio_tensors.append(audio_tensor)
            
        audio_tensors = torch.stack(audio_tensors)
        batch["audio_tensors"] = audio_tensors
        
        emotions_tensors = []
        for vid_name in videos:
            emotions_data = np.load('./emotion_features/{}.npy'.format(vid_name))
            emotions_tensor = torch.from_numpy(emotions_data).squeeze(0)
            emotions_tensors.append(emotions_tensor)
            

        emotions_tensors = torch.stack(emotions_tensors)
        
        batch["emotion_tensors"] = emotions_tensors


        return batch