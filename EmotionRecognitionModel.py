import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, BertModel, BertTokenizer, Wav2Vec2Model

#Defind the emotion model class
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.num_classes = num_classes

        # Load Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("auditi41/wav2vec2-large-xlsr-53-Bangla-Common_Voice")

        try:
            # Attempt to load the model
            self.wav2vec2_model = Wav2Vec2Model.from_pretrained("auditi41/wav2vec2-large-xlsr-53-Bangla-Common_Voice")
        except Exception as e:
            print(f"Error loading Wav2Vec2 model: {e}")
            self.wav2vec2_model = None

        # Load BERT model and tokenizer
        self.bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        # Classifier layers
        self.fc1 = nn.Linear(self.wav2vec2_model.config.hidden_size + self.bert_model.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, audio_inputs, input_ids, attention_mask):
        if self.wav2vec2_model is None:
            raise ValueError("Wav2Vec2 model is not properly loaded. Check for errors during model loading.")

        # Wav2Vec2 processing
        audio_outputs = self.wav2vec2_model(input_values=audio_inputs).last_hidden_state.squeeze(0)
        
        audio_features = torch.mean(audio_outputs, dim=1).squeeze(0)  # Pooling strategy: mean

        # BERT processing
        bert_outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[1]  # Extract pooled output

        # Concatenate audio and text features
        combined_features = torch.cat((audio_features, bert_outputs), dim=1)

        # Classification layers
        x = self.fc1(combined_features)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
