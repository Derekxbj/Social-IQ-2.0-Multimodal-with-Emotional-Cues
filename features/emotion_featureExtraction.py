import os
from datasets import Dataset

# create folder if not exists
if not os.path.exists("emotion_features"):
    os.makedirs("emotion_features")
        
        
from transformers import AutoTokenizer, T5Model, T5ForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

model = T5ForConditionalGeneration.from_pretrained("mrm8488/t5-base-finetuned-emotion")

import pandas as pd

matched_df_train = pd.read_json(path_or_buf= os.path.join('../../siq2/qa/qa_train.json'),lines=True)
matched_df_val = pd.read_json(path_or_buf= os.path.join('../../siq2/qa/qa_val.json'),lines=True)


import torch
import numpy as np

from datasets import Dataset
import pandas as pd
import os

import torch.nn.functional as F

for i,row in matched_df_train.iterrows():
    
    label = row["answer_idx"]
    
    # concatenate question and answer
    text = row["q"] + " " + row["a0"] + " " + row["a1"] + " " + row["a2"] + " " + row["a3"]
        
    # concatenation quesrion with each answer individually to four different sentences
    text = [row["q"] + " " + row["a0"] + '</s>', 
            row["q"] + " " + row["a1"] + '</s>', 
            row["q"] + " " + row["a2"] + '</s>', 
            row["q"] + " " + row["a3"] + '</s>']
    input_ids = tokenizer(text, return_tensors='pt', padding=True).input_ids
    
    
    with torch.no_grad():

        features = model.encoder(input_ids=input_ids).last_hidden_state.mean(dim=1)
        
    
    np.save("./emotion_features/{}.npy".format(row["vid_name"]), features.numpy())
    
    
for i,row in matched_df_val.iterrows():
    
    label = row["answer_idx"]
    
    # concatenate question and answer
    text = row["q"] + " " + row["a0"] + " " + row["a1"] + " " + row["a2"] + " " + row["a3"]
        
    # concatenation quesrion with each answer individually to four different sentences
    text = [row["q"] + " " + row["a0"] + '</s>', 
            row["q"] + " " + row["a1"] + '</s>', 
            row["q"] + " " + row["a2"] + '</s>', 
            row["q"] + " " + row["a3"] + '</s>']
    input_ids = tokenizer(text, return_tensors='pt', padding=True).input_ids
    
    
    with torch.no_grad():

        features = model.encoder(input_ids=input_ids).last_hidden_state.mean(dim=1)
        
    
    np.save("./emotion_features/{}.npy".format(row["vid_name"]), features.numpy())
    
