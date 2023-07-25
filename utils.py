from datasets import Dataset
import pandas as pd
import os


# get all file names in a foleder
def get_all_file_names(path):
    file_names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_names.append(file)
    return file_names

def get_videos_not_found(dataset_path, video_path):
    dataset = pd.read_json(path_or_buf= os.path.join(dataset_path),lines=True)
    
    all_videos = get_all_file_names(video_path)
    
    videos_not_found = []
    for i,row in dataset.iterrows():
        if (row["vid_name"]+'.mp4') not in all_videos:
            if row["vid_name"] not in videos_not_found:
                videos_not_found.append(row["vid_name"])
    
    return videos_not_found, all_videos
    
    
def show_one(example):
    print(f"Question: {example['q']}")
    print(f"  0 - {example['a0']} ")
    print(f"  1 - {example['a1']} ")
    print(f"  2 - {example['a2']} ")
    print(f"  3 - {example['a3']} ")
    print(f"\nGround truth: option {['0', '1', '2', '3'][example['answer_idx']]}")