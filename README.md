# Speech Emotion Recognition

This project performs speech emotion recognition using the RAVDESS dataset and audio feature extraction with `librosa`.

## Project Overview

The notebook `Speech_Emotion_Recognition_with_librosa.ipynb`:
- loads audio files from the RAVDESS dataset
- extracts features from each audio file:
  - MFCC
  - Chroma
  - Mel spectrogram
- trains an `MLPClassifier` from scikit-learn
- evaluates accuracy and F1 score
- saves the trained model with `pickle`
- loads the model and makes predictions on new audio files

## Dataset

The project expects the RAVDESS speech emotion dataset in:

`dataset/speech-emotion-recognition-ravdess-data`

Audio files are located under:
`Actor_*/ *.wav`

## Emotions used

The notebook selects a subset of emotions:

- calm
- happy
- fearful
- disgust

## Requirements

- Python 3.x
- librosa
- soundfile
- numpy
- scikit-learn
- pandas
- pickle (built-in)

## Usage

1. Install dependencies:
   ```bash
   pip install librosa soundfile numpy scikit-learn pandas
2. Set the dataset root path in the notebook, or update the hardcoded paths.
3. Run the notebook cells in order:

    - extract features
    - load data
    - split dataset
    - train model
    - evaluate model
    - save model
    - predict on new audio
# Feature extraction

The notebook uses the extract_feature function:

  - reads audio with soundfile
  - computes MFCC, chroma, and mel spectrogram features
  - concatenates them into one feature vector
# Model training

MLPClassifier is configured with:

 . alpha=0.01
 . batch_size=256
 . epsilon=1e-08
 . hidden_layer_sizes=(300,)
 . learning_rate='adaptive'
 . max_iter=500

# Training and test split:

 . test_size=0.25

# Notes
 - Paths are currently hardcoded for Windows.
 - Update the dataset path if your data is in a different folder.
 - The notebook shows evaluation with accuracy and F1 score.