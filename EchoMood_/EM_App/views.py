from django.shortcuts import render
from django.core.files.storage import default_storage
import pickle
import os
import librosa
import soundfile
import numpy as np
from django.shortcuts import render
from .forms import AudioForm
from django.core.files.base import ContentFile
from.models import AudioPrediction
from pydub import AudioSegment

# Define the path to your model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_det_model.pkl')

# Load the model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)


def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    new_file_path = file_path.replace(".wav", "_converted.wav")
    audio.export(new_file_path, format="wav")
    return new_file_path


def extract_features(file_name, mfcc=True, chroma=True, mel=True):
    try:
        audio_data, sample_rate = librosa.load(file_name, sr=None)

        if mfcc:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            mfccs = np.mean(mfccs.T, axis=0)

        if chroma:
            stft = np.abs(librosa.stft(audio_data))
            chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            chroma = np.mean(chroma.T, axis=0)

        if mel:
            mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
            mel = np.mean(mel.T, axis=0)

        result = np.hstack([mfccs, chroma, mel])
        return result

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None




EMOTION_GIFS = {
    'calm': """
        <div class="tenor-gif-embed" data-postid="16137769" data-share-method="host" data-aspect-ratio="1.33891" data-width="50%">
            <a href="https://tenor.com/view/chill-zen-buddha-omm-spongebob-squarepants-gif-16137769">Chill Zen GIF</a> from 
            <a href="https://tenor.com/search/chill-gifs">Chill GIFs</a>
        </div> 
        <script type="text/javascript" async src="https://tenor.com/embed.js"></script>
    """,
    'fearful': """
        <div class="tenor-gif-embed" data-postid="7935022" data-share-method="host" data-aspect-ratio="1.785" data-width="50%">
            <a href="https://tenor.com/view/sweating-nervous-scared-key-and-peele-gif-7935022">Sweating Nervous GIF</a> from 
            <a href="https://tenor.com/search/sweating-gifs">Sweating GIFs</a>
        </div> 
        <script type="text/javascript" async src="https://tenor.com/embed.js"></script>
    """,
    'happy': """
        <div class="tenor-gif-embed" data-postid="25226897" data-share-method="host" data-aspect-ratio="0.81875" data-width="50%">
            <a href="https://tenor.com/view/happy-dance-gif-25226897">Happy Dance GIF</a> from 
            <a href="https://tenor.com/search/happy-gifs">Happy GIFs</a>
        </div> 
        <script type="text/javascript" async src="https://tenor.com/embed.js"></script>
    """,
    'disgust': """
        <div class="tenor-gif-embed" data-postid="4990489" data-share-method="host" data-aspect-ratio="0.906832" data-width="50%">
            <a href="https://tenor.com/view/wow-weird-skeptical-worried-disgusted-gif-4990489">Wow Weird GIF</a> from 
            <a href="https://tenor.com/search/wow-gifs">Wow GIFs</a>
        </div> 
        <script type="text/javascript" async src="https://tenor.com/embed.js"></script>
    """,
    # Add other emotions and corresponding GIFs here
}




def predict_emotion(request):
    gif_embed_code = '' 
    predicted_emotion = ''
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES)
        if form.is_valid():
            audio_file = form.cleaned_data['audio_file']
            audio_file_name = default_storage.save(audio_file.name, ContentFile(audio_file.read()))
            audio_file_url = default_storage.url(audio_file_name)  # Get the URL of the uploaded audio file
            
            converted_file_path = convert_to_wav(default_storage.path(audio_file_name))
            

            features = extract_features(converted_file_path, mfcc=True, chroma=True, mel=True)  # Implement this function
            if features is not None:
                # Ensure the features are in the correct 2D shape for model prediction
                features = np.array(features).reshape(1, -1)
                predicted_emotion = model.predict(features)[0]

                # Save the prediction to the database
                prediction = AudioPrediction.objects.create(
                    audio_file=audio_file_name,
                    predicted_emotion=predicted_emotion
                )

                # Get the corresponding GIF embed code for the predicted emotion
                gif_embed_code = EMOTION_GIFS.get(predicted_emotion, '')

                # Get the URL of the uploaded audio file
                audio_file_url = default_storage.url(audio_file_name)
                
                return render(request, 'result.html', {
                    'gif_embed_code': gif_embed_code,
                    'emotion': predicted_emotion,
                    'audio_file_url': audio_file_url
                })

            return render(request, 'result.html', {'gif_embed_code': gif_embed_code, 'emotion': predicted_emotion,'audio_file_url': audio_file_url})
    else:
        form = AudioForm()
    return render(request, 'predict.html', {'form': form})