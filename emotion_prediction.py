import torch
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from model import ParallelModel  # Import the model from model.py

# Emotion mapping
EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}

def preprocess_audio(audio_path, sample_rate=16000, target_length=16000):
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        pad = target_length - len(audio)
        audio = np.pad(audio, (0, pad))
    return audio

def getMELspectrogram(audio, sample_rate=16000, fixed_time=563):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate,
                                              n_fft=1024,
                                              win_length=512,
                                              window='hamming',
                                              hop_length=256,
                                              n_mels=128,
                                              fmax=sample_rate/2)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Ensure mel spectrogram has fixed size (128, fixed_time)
    if mel_spec_db.shape[1] > fixed_time:
        mel_spec_db = mel_spec_db[:, :fixed_time]
    elif mel_spec_db.shape[1] < fixed_time:
        pad_width = fixed_time - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')

    return mel_spec_db

def predict_emotion(model, scaler, audio_path, device='cpu', sample_rate=16000):
    signal = preprocess_audio(audio_path, sample_rate=sample_rate)
    mel_spectrogram = getMELspectrogram(signal, sample_rate=sample_rate)
    
    print("Shape of mel_spectrogram before scaling:", mel_spectrogram.shape)
    
    mel_spectrogram = mel_spectrogram[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, 128, 563)
    
    b, c, h, w = mel_spectrogram.shape
    mel_spectrogram = np.reshape(mel_spectrogram, newshape=(b, -1))
    
    print("Shape of mel_spectrogram reshaped for scaling:", mel_spectrogram.shape)
    
    mel_spectrogram = scaler.transform(mel_spectrogram)
    mel_spectrogram = np.reshape(mel_spectrogram, newshape=(b, c, h, w))
    
    print("Shape of mel_spectrogram after scaling:", mel_spectrogram.shape)
    
    mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32).to(device)
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output_logits, output_softmax, attention_weights_norm = model(mel_spectrogram)
        predicted_emotion = torch.argmax(output_softmax, dim=1).item()
    
    return predicted_emotion

# Load your trained model and scaler
model = ParallelModel(num_emotions=8)  # Replace 8 with the actual number of emotions
model.load_state_dict(torch.load('cnn_lstm_parallel_model.pt', map_location=torch.device('cpu')))

scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')  # Load the scaler mean
scaler.scale_ = np.load('scaler_scale.npy')  # Load the scaler scale

# Print shapes of loaded scaler parameters
print("Shape of scaler mean:", scaler.mean_.shape)
print("Shape of scaler scale:", scaler.scale_.shape)

# Path to the audio file you want to predict
audio_path = '03-01-07-02-01-01-24-disgust.wav'

# Predict the emotion
predicted_emotion = predict_emotion(model, scaler, audio_path)
print(f'Predicted emotion: {predicted_emotion}')


# 03-01-08-02-01-01-24-surprised
# 03-01-07-02-01-01-24-disgust
# 03-01-06-02-01-01-24-fearful
# 03-01-05-02-01-02-24-angry
# 03-01-04-02-02-01-24-sad
# 03-01-03-02-02-01-24-happy
# 03-01-02-02-02-02-24-calm
# 03-01-01-01-02-02-24-neutral
