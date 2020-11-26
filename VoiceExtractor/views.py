from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm

from io import BytesIO
import numpy as np
import scipy.io.wavfile as wavfile
from .models import VoiceExtractor


def upload(request):
    content=[]
    if request.method == 'POST':
        print(request.FILES)
        noisy_train = request.FILES['noisy_train'].file
        clear_train = request.FILES['clear_train'].file
        noisy_file = request.FILES['noisy_file'].file
        handle_data(noisy_train, clear_train, noisy_file)
    return render(request, 'upload.html', {'content':content})


def handle_data(noisy_train, clear_train, noisy_file):
    voice_extractor = VoiceExtractor()
    voice_extractor.load_data(noisy_train, clear_train)
    voice_extractor.create_model()
    voice_extractor.compile_model(1e-2)
    voice_extractor.fit_model(steps=1330, epochs=20, validation_split=0.05)

    voice_extractor.filter_file(noisy_file, 'clear_file.wav')
    return None
