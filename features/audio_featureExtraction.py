import os
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model


# create folder if not exists
if not os.path.exists("audio_features"):
    os.makedirs("audio_features")

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# iterate over all audio files
for audio in os.listdir("../../siq2/audio/wav"):
    print("audio: ", audio)
    vid_name = audio.split(".")[0]
    
    # load audio
    waveform, sample_rate = torchaudio.load('../../siq2/audio/wav/{}.wav'.format(vid_name))
    
    # get the lenght of the audio
    audio_length = waveform.shape[1]

    # downsample audio to 16kHz
    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    audio_length = waveform.shape[1]
    
    # split the audio length into 21 evenly spaced timestamps
    timestamps = np.linspace(0, audio_length, 21)
    
    audio_tensors = []
    for i in range(len(timestamps)):
        if i==0:
            continue
        else:
            start_sec = timestamps[i-1]
            end_sec = timestamps[i]
            audio_data = waveform[:, int(start_sec):int(end_sec)]
            audio_features = processor(audio_data, sampling_rate=16000, return_tensors="pt")
            audio_tensors.append(audio_features.input_values.squeeze(0).permute(1, 0))
    
    # pad audio tensors to same length
    audio_tensors = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True, padding_value=0)
    
    audio_tensors = audio_tensors.squeeze(2)
    
    # save features to disk
    np.save("./audio_features/{}.npy".format(vid_name), audio_tensors.numpy())
    