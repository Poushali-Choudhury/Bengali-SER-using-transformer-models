import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib
from torchaudio.transforms import Resample, Vol
import torch.multiprocessing as multiprocessing

from EmotionRecognitionModel import EmotionRecognitionModel

# Define constants
audio_path = "/python_proj/mixed/"
num_classes = 7
max_audio_length = 60000
max_text_length = 100
batch_size = 10
learning_rate = 1e-5
epochs = 12

# List to store audio files and emotion labels
audio_files = []
emotion_labels = []
label_encoder = LabelEncoder()

# Load audio files and extract emotion labels
for file_name in os.listdir(audio_path):
    if file_name.endswith(".wav"):
        audio_files.append(os.path.join(audio_path, file_name))
        emotion_label = file_name.split("_")[-2]
        emotion_labels.append(emotion_label)

# Split the data into train and validation sets
train_paths, val_paths, train_labels, val_labels = train_test_split(audio_files, emotion_labels, test_size=0.2,
                                                                    random_state=42)

# Fit label encoder on training set only
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)

# Transform both train and validation labels
train_labels_encoded = label_encoder.transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)

# Define a dataset class for audios
class AudioDataset(Dataset):
    def __init__(self, audio_paths, emotion_labels, wav2vec2_processor, bert_tokenizer, max_audio_length,
                 max_text_length, label_encoder):
        self.audio_paths = audio_paths
        self.labels = emotion_labels
        self.wav2vec2_processor = wav2vec2_processor
        self.bert_tokenizer = bert_tokenizer
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        try:
            waveform, sample_rate = torchaudio.load(audio_path, normalize=True) # loading audios

            waveform = torchaudio.transforms.Vad(sample_rate=sample_rate)(waveform) # removing silence from front and end
            waveform = Vol(gain=2.0)(waveform) # normalizing

            if sample_rate != 16000:
                waveform = Resample(orig_freq=sample_rate, new_freq=16000)(waveform) # resampling to 16000

        except Exception as e:
            print(f"Error loading audio file: {audio_path}, error: {str(e)}")
            # Handle the error appropriately
            waveform = torch.zeros((1, 1))  # Placeholder waveform

        # pre processing with wav2vec2 2.0
        inputs = self.wav2vec2_processor(waveform.squeeze().numpy(), return_tensors="pt", padding='max_length',
                                         truncation=True, max_length=self.max_audio_length, sampling_rate=16000)

        # Convert speech to text using Wav2Vec2 model
        with torch.no_grad():
            logits = wav2vec2_model(input_values=inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcriptions = self.wav2vec2_processor.batch_decode(predicted_ids)

        # Tokenize transcriptions using BERT tokenizer
        tokenized_inputs = self.bert_tokenizer(transcriptions, max_length=self.max_text_length, padding='max_length',
                                                truncation=True, return_tensors="pt", return_attention_mask=True)

        input_ids = tokenized_inputs.input_ids.squeeze(0)
        attention_mask = tokenized_inputs.attention_mask.squeeze(0)
        audio_inputs = inputs.input_values.squeeze(0)

        # Encode labels
        encoded_label = torch.tensor(self.label_encoder.transform([label])[0], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_inputs": audio_inputs,
            "labels": encoded_label
        }

# Load Wav2Vec2 model and tokenizer
wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("auditi41/wav2vec2-large-xlsr-53-Bangla-Common_Voice")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("auditi41/wav2vec2-large-xlsr-53-Bangla-Common_Voice")

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Prepare dataset
train_dataset = AudioDataset(train_paths, train_labels, wav2vec2_processor, bert_tokenizer, max_audio_length,
                             max_text_length, label_encoder)
val_dataset = AudioDataset(val_paths, val_labels, wav2vec2_processor, bert_tokenizer, max_audio_length,
                           max_text_length, label_encoder)

#Prepare dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)


joblib.dump(label_encoder, 'label_encoder_m.pkl') # dumping the label encoder

model = EmotionRecognitionModel(num_classes) # load the adapted model with 7 emotions

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

#defined main function
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn') 
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            audio_inputs = batch["audio_inputs"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()

            outputs = model(audio_inputs, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )


        # Evaluation loop for the model on the validation set
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation"):
                audio_inputs = batch["audio_inputs"]
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # Forward pass
                outputs = model(audio_inputs, input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Decode labels and predictions
                _, predicted = torch.max(outputs, 1)
                val_targets.extend(labels.tolist())
                val_predictions.extend(predicted.tolist())

        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_f1 = f1_score(val_targets, val_predictions, average="weighted")
        val_precision_score = precision_score(val_targets, val_predictions, average='weighted')
        val_recall_score = recall_score(val_targets, val_predictions, average='weighted')
        val_confusion_m = confusion_matrix(val_targets, val_predictions)

        print(
            f"Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}; "
            f"Validation Precision score: {val_precision_score:.4f}, Validation Recall: {val_recall_score:.4f}"
        )

        print("Confusion Matrix:")
        print(val_confusion_m)

    # Save the trained model with information
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, "emotion_recognition_model_checkpoint_m.pth")
