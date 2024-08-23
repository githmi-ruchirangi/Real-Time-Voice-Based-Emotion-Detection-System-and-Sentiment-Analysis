import torch
import torch.nn as nn
import librosa
import numpy as np
from model import ParallelModel  # Assuming this imports your model correctly

# Emotion mapping
EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}

# Function to extract Mel spectrogram
def getMELspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                            sr=sample_rate,
                                            n_fft=1024,
                                            win_length=512,
                                            window='hamming',
                                            hop_length=256,
                                            n_mels=128,
                                            fmax=sample_rate/2)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Function to prepare input data
def prepare_input(audio_file, sample_rate):
    # Load audio file
    audio, _ = librosa.load(audio_file, sr=sample_rate, duration=3.0)
    # Extract features (Mel spectrogram)
    mel_spectrogram = getMELspectrogram(audio, sample_rate)
    # Convert to numpy array and reshape to match model input (if necessary)
    # Example assumes a single channel spectrogram, adjust as per your model
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add batch dimension if needed
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Add channel dimension if needed
    # Convert numpy array to PyTorch tensor
    mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
    return mel_spectrogram

# Load and use the trained model for prediction
def predict_emotion(audio_file, model_path):
    # Define model and load trained parameters
    model = ParallelModel(num_emotions=8)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare input data
    sample_rate = 48000  # Replace with your audio file's sample rate
    input_data = prepare_input(audio_file, sample_rate)

    # Perform prediction
    with torch.no_grad():
        output_logits, output_softmax, _ = model(input_data)

    # Get predicted emotion
    _, predicted_emotion = torch.max(output_softmax, 1)
    return EMOTIONS[predicted_emotion.item()]

# Example usage
if __name__ == "__main__":
    audio_file = 'sad.wav'
    model_path = 'cnn_lstm_parallel_model.pt'

    predicted_emotion = predict_emotion(audio_file, model_path)
    print("Emotion:", predicted_emotion)
