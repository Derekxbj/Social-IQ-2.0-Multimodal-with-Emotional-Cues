import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import myModel
from data_load import generateDataset, DataCollatorForMultipleChoice

device = torch.device("cuda:0")


path = "PATH_TO_MODEL"

tokenizer = AutoTokenizer.from_pretrained(path)


dataset = generateDataset('../siq2/qa/qa_train.json', '../siq2/qa/qa_test.json', '../siq2/video', tokenizer)
tokenized_train, tokenized_val = dataset.returnDataset()

import pandas as pd
df_test = pd.read_json(path_or_buf= os.path.join('./qa_test_text.json'),lines=True)


model = myModel.from_pretrained('roberta-large')
state_dict = torch.load(os.path.join(path, 'model_roberta-large.pth'))
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") # remove ".module" from key
    new_state_dict[name] = v


model.load_state_dict(new_state_dict)
print("model loaded!")

model.to(device)

test_set = DataLoader(tokenized_val, 
                        batch_size=1, 
                        shuffle=False,
                        num_workers=0,
                        pin_memory=True,
                        collate_fn=DataCollatorForMultipleChoice(tokenizer))


from torchmetrics.classification import Accuracy
from tqdm import tqdm


acc_metric = Accuracy(task="multiclass", num_classes=4)
acc_metric.to(device)

    
counts = 0 

for batch in tqdm(test_set):
    
    id = batch.pop("sample_ids")
    batch = {k:v.to(device) for k,v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        
        _, preds = torch.max(logits, 1)
        
            
    # assign the answer_idx column to the predicted answer
    df_test.loc[df_test['qid'] == id[0], 'answer_idx'] = int(preds)



print("counts", counts)

# save the dataframe as a json file in current directory
df_test.to_json(path_or_buf= os.path.join('./', 'qa_test.json'), orient='records', lines=True)