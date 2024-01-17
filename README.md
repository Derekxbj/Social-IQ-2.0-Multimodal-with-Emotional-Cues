# Social-IQ-2.0-Multimodal-with-Emotional-Cues

<!-- # Multi-Modal Correlated Network with Emotional Reasoning Knowledge for Social Intelligence Question-Answering -->

This repository is the implementation of [Multi-Modal Correlated Network with Emotional Reasoning Knowledge for Social Intelligence Question-Answering](https://openaccess.thecvf.com/content/ICCV2023W/ASI/papers/Xie_Multi-Modal_Correlated_Network_with_Emotional_Reasoning_Knowledge_for_Social_Intelligence_ICCVW_2023_paper.pdf). 

## Requirements

Please follow the instructions of the [Social-IQ 2.0 benchmark](https://github.com/abwilf/Social-IQ-2.0-Challenge) to download the dataset and the required dependecies.

Then, to install requirements of this repository:

```setup
pip install -r requirements.txt
```

## Feature Extraction

To extract the features, run the follwing commands in the features folder:

```feature extraction
python audio_featureExtraction.py
python video_featureExtraction.py
python emotion_featureExtraction.py
```


## Training

To train the model in the paper, run this command:

```train
python -m torch.distributed.launch --nproc_per_node=4 main.py --ngpu 4
```

## Evaluation

To evaluate the model in the paper, run this command:

```train
python evaluation.py
```


