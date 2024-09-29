import torch.nn as nn
from transformers import BertModel


class EmotionCauseModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_emotions=6, hidden_size=768):
        super(EmotionCauseModel, self).__init__()

        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Emotion classification head (Softmax over emotion labels)
        self.emotion_classifier = nn.Linear(hidden_size, num_emotions)

        # AYOUB
        # Cause identification head (Attention over previous utterances)
        self.cause_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.cause_classifier = nn.Linear(hidden_size, 1)  # Binary cause/no-cause prediction

        # Activation function
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, token_type_ids=None, causal_input_ids=None,
                causal_attention_mask=None):
        # BERT encoding for current utterance
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Emotion classification
        emotion_logits = self.emotion_classifier(sequence_output[:, 0, :])  # CLS token
        emotion_probs = self.softmax(emotion_logits)

        # Cause identification (attend to past utterances)
        if causal_input_ids is not None:
            causal_outputs = self.bert(input_ids=causal_input_ids, attention_mask=causal_attention_mask)
            causal_sequence_output = causal_outputs.last_hidden_state

            # Apply attention between the current utterance and all previous utterances
            attn_output, _ = self.cause_attention(sequence_output, causal_sequence_output, causal_sequence_output)

            # Classify cause/no-cause for each token in past utterances
            cause_logits = self.cause_classifier(attn_output).squeeze(-1)
            cause_probs = self.sigmoid(cause_logits)
            return emotion_probs, cause_probs
        else:
            return emotion_probs
