from django.db import models

# Create your models here.
class AudioPrediction(models.Model):
    audio_file=models.FileField(upload_to='audio_files/')
    predicted_emotion=models.CharField(max_length=100)
    uploaded_at=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_emotion}on {self.uploaded_at}"