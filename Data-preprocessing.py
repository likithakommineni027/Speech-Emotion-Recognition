# ✅ Updated full pipeline: Extract & save sequential features (no flattening)

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Emotion mapping dictionaries
emotion_map_ravdess = {
    1: "neutral", 2: "neutral", 3: "happy", 4: "sad", 5: "angry",
    6: "fear", 7: "disgust", 8: "surprise"
}

emotion_map_tess = {
    'OAF_Fear': 'fear', 'OAF_Pleasant_surprise': 'surprise', 'OAF_Sad': 'sad', 'OAF_angry': 'angry',
    'OAF_disgust': 'disgust', 'OAF_happy': 'happy', 'OAF_neutral': 'neutral',
    'YAF_angry': 'angry', 'YAF_disgust': 'disgust', 'YAF_fear': 'fear',
    'YAF_happy': 'happy', 'YAF_neutral': 'neutral', 'YAF_pleasant_surprise': 'surprise', 'YAF_sad': 'sad'
}

emotion_dict_emodb = {
    'W': 'angry', 'L': 'neutral', 'A': 'fear', 'F': 'happy',
    'T': 'sad', 'E': 'disgust', 'N': 'neutral'
}

def add_white_noise(signal, snr_db=20):
    noise_power = np.mean(signal ** 2) / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

def shift_signal(signal, shift_max=0.25, sampling_rate=16000, shift_direction='right'):
    shift = np.random.randint(0, int(sampling_rate * shift_max))
    return np.roll(signal, -shift if shift_direction == 'right' else shift)

def extract_features(y, sr=16000, hop_length=512):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    stacked = np.vstack([mfccs, log_mel_spec, zcr, rms, chroma]).T  # shape: (T, F)
    return stacked.astype(np.float32)

def load_dataset(dataset_path, dataset_type):
    file_paths, labels = [], []
    if dataset_type == 'ravdess':
        for actor_folder in os.listdir(dataset_path):
            actor_path = os.path.join(dataset_path, actor_folder)
            if os.path.isdir(actor_path):
                for filename in os.listdir(actor_path):
                    if filename.endswith('.wav'):
                        emotion_code = int(filename.split('-')[2])
                        emotion = emotion_map_ravdess.get(emotion_code, "unknown")
                        file_paths.append(os.path.join(actor_path, filename))
                        labels.append(emotion)
    elif dataset_type == 'tess':
        for subfolder in os.listdir(dataset_path):
            subfolder_path = os.path.join(dataset_path, subfolder)
            if os.path.isdir(subfolder_path):
                emotion = emotion_map_tess.get(subfolder, "unknown")
                if emotion == "unknown":
                    continue
                for filename in os.listdir(subfolder_path):
                    if filename.endswith('.wav'):
                        file_paths.append(os.path.join(subfolder_path, filename))
                        labels.append(emotion)
    elif dataset_type == 'emodb':
        for filename in os.listdir(dataset_path):
            if filename.endswith('.wav'):
                emotion_label = filename[5]
                emotion = emotion_dict_emodb.get(emotion_label, "unknown")
                file_paths.append(os.path.join(dataset_path, filename))
                labels.append(emotion)
    return file_paths, labels

def process_and_augment(file_paths, labels, label_encoder, fixed_length=16000):
    signals, encoded_labels = [], []
    for file_path, label in zip(file_paths, labels):
        try:
            signal, sr = librosa.load(file_path, sr=16000, duration=5.0)
            signal = np.pad(signal, (0, max(0, fixed_length - len(signal))))[:fixed_length]

            augmentations = [
                signal,
                add_white_noise(signal),
                shift_signal(signal, sampling_rate=sr),
                librosa.effects.time_stretch(signal, rate=0.9),
                librosa.effects.time_stretch(signal, rate=1.1),
                librosa.effects.pitch_shift(signal, sr=sr, n_steps=2),
                librosa.effects.pitch_shift(signal, sr=sr, n_steps=-2)
            ]

            for aug in augmentations:
                aug = np.pad(aug, (0, max(0, fixed_length - len(aug))))[:fixed_length]
                signals.append(aug.astype(np.float32))
                encoded_labels.append(label_encoder.transform([label])[0])
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return np.array(signals), np.array(encoded_labels)

def extract_features_from_signals(signals, sr=16000):
    return [extract_features(signal, sr) for signal in signals]

# Load and combine datasets
DATASET_PATHS = {
    'ravdess': 'RAVDESS',
    'tess': 'TESS',
    'emodb': 'EMODB'
}

all_file_paths = []
all_labels = []

print("\n🔄 Loading all datasets and combining...")
for dataset_name, dataset_path in DATASET_PATHS.items():
    file_paths, labels = load_dataset(dataset_path, dataset_name)
    if not file_paths:
        print(f"⚠️ No valid files found in {dataset_name.upper()}! Skipping...")
        continue
    all_file_paths.extend(file_paths)
    all_labels.extend(labels)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Process + augment
print("🎧 Processing and augmenting combined dataset...")
signals, encoded_labels = process_and_augment(all_file_paths, all_labels, label_encoder)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(signals, encoded_labels, test_size=0.25, random_state=42)

# Extract features from audio
print("🔍 Extracting features...")
train_features_raw = extract_features_from_signals(X_train)
test_features_raw = extract_features_from_signals(X_test)

# Pad sequences to uniform length
max_len = max(max([f.shape[0] for f in train_features_raw]), max([f.shape[0] for f in test_features_raw]))
train_features = pad_sequences(train_features_raw, maxlen=max_len, dtype='float32', padding='post', truncating='post')
test_features = pad_sequences(test_features_raw, maxlen=max_len, dtype='float32', padding='post', truncating='post')

# Save
np.save('train_features_combined.npy', train_features)
np.save('test_features_combined.npy', test_features)
np.save('train_labels_combined.npy', y_train)
np.save('test_labels_combined.npy', y_test)
print("\n✅ Sequential feature dataset saved for CNN + BiLSTM!")
