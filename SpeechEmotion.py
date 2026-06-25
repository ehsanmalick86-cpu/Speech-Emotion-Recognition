import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# --- STEP 1: READ SOUND FILES AND EXTRACT NUMBERS ---
def extract_numbers_from_sound(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Emotions mapped to RAVDESS code numbers
emotions_map = {'01':'neutral', '03':'happy', '04':'sad', '05':'angry'}

X = [] 
y = [] 

folder_path = "./audio_data/Actor_*/*.wav"
print("Looking for audio files...")

for file in glob.glob(folder_path):
    file_name = os.path.basename(file)
    emotion_code = file_name.split('-')[2] 
    
    if emotion_code in emotions_map:
        sound_features = extract_numbers_from_sound(file)
        X.append(sound_features)
        y.append(emotions_map[emotion_code])

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("\n[ERROR] No audio files found! Check folder path.")
else:
    print(f"Successfully loaded {len(X)} audio files.")

    # --- STEP 2: SPLIT DATA ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- STEP 3: BUILD AND TRAIN AI ---
    print("Training the AI model now... Please wait...")
    model = MLPClassifier(alpha=0.01, batch_size=16, hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)

    # --- STEP 4: TEST THE AI ---
    predictions = model.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=predictions)
    
    print(f"\nFinal AI Accuracy Score: {score * 100:.2f}%\n")
    
    # --- NEW DETAILED PRINT OUT ---
    print("--- Detailed Guess List ---")
    for i in range(len(X_test)):
        real_emotion = y_test[i]
        ai_guess = predictions[i]
        
        if real_emotion == ai_guess:
            print(f"Audio {i+1}: AI guessed right! -> [ {ai_guess.upper()} ]")
        else:
            print(f"Audio {i+1}: AI Mistake! Real: {real_emotion} | Guess: {ai_guess}")