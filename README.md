# Bengali-SER-using-transformer-models
## Project Title: 
    Bengali Speech Emotion recognition using Wav2Vec2 and BERT model
## Description: 
    Determining Emotions from Bengali speech using Wav2Vec2 and BERT model to extract both langugage dependent and independent features. 
## Installation: 
    The following steps are required to run the program: 

    1. Need to install below mentioned libraries on the python computing platform: 
      	numpy
      	torch
      	torchaudio
      	scikit-learn
      	transformers
      	matplotlib
      	seaborn
      	IPython
      	pydub
      	pandas
    
    2. update datasets path and import EmotionRecognitionModel
    3. run the tranandval.py 

## Data: 
    Datasets are used in this projects are a mixed audio files from SUBESCO(SUST Bangla Emotional Speech Corpus) and BASER(Bangla Adolescent Speech Emotion Recognition)           datasets. Which are availble online for free.
    https://doi.org/10.1371/journal.pone.0250173  (SUBESCO)
    https://www.kaggle.com/datasets/jabedcse/baser-dataset (BASER)


## Models: 
    Pre-trained models are used from hugging face
    Wav2Vec2: auditi41/wav2vec2-large-xlsr-53-Bangla-Common_Voice
    BERT: bert-base-multilingual-cased
## Evaluation: 
    Metrices are used Accuracy, Precision, Recall, F1 Score, Confusion Matrix
## Results: 
    After training, the model achieves an accuracy of 82.86% on the validation set, with high precision and recall for most emotion classes.
