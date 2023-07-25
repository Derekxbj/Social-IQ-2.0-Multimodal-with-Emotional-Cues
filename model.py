import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import VideoMAEModel

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForMultipleChoice, RobertaPreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from loss import InfoNCE

_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1, x2, x3):
        attention_output, _ = self.attention(x1, x2, x3)
        attention_output = self.layer_norm1(x1 + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(attention_output)
        output = self.layer_norm2(attention_output + self.dropout(feed_forward_output))
        return output


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, size_average=True):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        if size_average:
            loss_contrastive = torch.mean(loss_contrastive)

        return loss_contrastive

@add_start_docstrings(
    """
    Roberta Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_START_DOCSTRING,
)
class myModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, 1)
        
        
        # Initialize weights and apply final processing
        self.post_init()
        

        # video model
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics", output_hidden_states=True)
        
        # audio model
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)
        
        self.mha_video = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=4, batch_first=True)
        self.linear_video = nn.Linear(768, 1024)
        
        self.mha_audio = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=4, batch_first=True)
        self.linear_audio = nn.Linear(768, 1024)
        
        
        self.contrastive_loss = InfoNCE(negative_mode='paired')
        

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        video_tensors: Optional[torch.FloatTensor] = None,
        audio_tensors: Optional[torch.FloatTensor] = None,
        emotion_tensors: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1] # (batch_size * num_choices, hidden_size)
        
        reshaped_pooled_output = pooled_output.view(-1, num_choices, pooled_output.size(-1)) # (batch_size, num_choices, hidden_size)
        
        
        # video features
        video_outputs = []
        for i in range(video_tensors.shape[1]):
            with torch.no_grad():
                video_outputs.append(self.videomae(video_tensors[:, i]).last_hidden_state[:,0]) # (batch_size, num_frames, hidden_size)
        
        video_outputs = torch.stack(video_outputs, dim=1) # (batch_size, num_frames, hidden_size)
        video_outputs = self.linear_video(video_outputs)

        
        # audio features
        audio_outputs = []
        for i in range(audio_tensors.shape[1]):
            with torch.no_grad():
                audio_outputs.append(self.wav2vec(audio_tensors[:, i]).last_hidden_state[:,0]) # (batch_size, num_frames, hidden_size)
        audio_outputs = torch.stack(audio_outputs, dim=1)
        audio_outputs = self.linear_audio(audio_outputs)

        
        video_attn, _ = self.mha_video(reshaped_pooled_output, video_outputs, video_outputs) # (batch_size, num_choices, hidden_size)
        video_attn = video_attn.reshape(-1, video_attn.size(-1)) # (batch_size * num_choices, hidden_size)
        
        audio_attn, _ = self.mha_audio(reshaped_pooled_output, audio_outputs, audio_outputs) # (batch_size, num_choices, hidden_size)
        audio_attn = audio_attn.reshape(-1, audio_attn.size(-1)) # (batch_size * num_choices, hidden_size)
        
        
        
        # cat the video and audio features to the pooled output
        output = torch.cat((pooled_output, video_attn, audio_attn), dim=1) # (batch_size * num_choices, hidden_size * 2)

        output = self.dropout(output)
        logits = self.classifier(output) # (batch_size * num_choices, 1)
        reshaped_logits = logits.view(-1, num_choices) # (batch_size, num_choices)
        

        loss = None
        if labels is not None:

            # move labels to correct device to enable model parallelism
            labels = labels.to(reshaped_logits.device) # (batch_size)
            loss_fct = CrossEntropyLoss()
            # loss = loss_fct(reshaped_logits, labels)
            
            emotion_correct = emotion_tensors[torch.arange(emotion_tensors.size(0)), labels].unsqueeze(1)
            emotion_similarity = F.cosine_similarity(emotion_correct, emotion_tensors, dim=2)
            distractor_indexs = torch.topk(emotion_similarity, largest=False, k=2, dim=1).indices

            
            # pooled_output = pooled_output.view(-1, num_choices, pooled_output.size(-1))
            video_attn = video_attn.view(-1, num_choices, video_attn.size(-1))
            query_video = video_attn[torch.arange(video_attn.size(0)), labels]
            audio_attn = audio_attn.view(-1, num_choices, audio_attn.size(-1))
            query_audio = audio_attn[torch.arange(audio_attn.size(0)), labels]
            
            
            positive = reshaped_pooled_output[torch.arange(reshaped_pooled_output.size(0)), labels]
            # find the top 2 distractors from pooled_output along the batch dimension usint index shape (batch_size, 2)
            negative_one = reshaped_pooled_output[torch.arange(reshaped_pooled_output.size(0)), distractor_indexs[:,0]]
            negative_two = reshaped_pooled_output[torch.arange(reshaped_pooled_output.size(0)), distractor_indexs[:,1]]
            # concat negative_one and negative_two using a new dimension 
            negatives = torch.stack((negative_one, negative_two), dim=1)
            
            
            loss_v = self.contrastive_loss(query_video, positive, negatives)
            loss_a = self.contrastive_loss(query_audio, positive, negatives)
            
            loss = loss_fct(reshaped_logits, labels) + 0.1*(loss_v + loss_a)/2
            

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )