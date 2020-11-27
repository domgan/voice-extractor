from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from .forms import UploadFileForm

from io import BytesIO
import numpy as np
import scipy.io.wavfile as wavfile
from .models import VoiceExtractor
import tempfile
from django.http import HttpResponse, Http404
import os
import tempfile

def upload(request):
    content=[]
    if request.method == 'POST':
        noisy_train = request.FILES['noisy_train'].file
        clear_train = request.FILES['clear_train'].file
        model_weights = create_model(noisy_train, clear_train)
        model_weights = [weights.tolist() for weights in model_weights]  # change list of ndarrays to list of list for json serialization
        request.session['model_weights'] = model_weights
        return redirect('/filter/')
    return render(request, 'train.html', {'content':content})


def filter(request):
    content=[]
    if request.method == 'POST':
        noisy_file = request.FILES['noisy_file'].file
        model_weights = [np.array(weights) for weights in request.session['model_weights']]
        filter_file(noisy_file, model_weights, request)
    return render(request, 'filter.html', {'content':content})


def create_model(noisy_train, clear_train):
    voice_extractor = VoiceExtractor()
    voice_extractor.load_data(noisy_train, clear_train)
    voice_extractor.create_model()
    voice_extractor.compile_model(1e-2)
    voice_extractor.fit_model(steps=665, epochs=10, validation_split=0.05)  # 1330
    model_weights = voice_extractor.model.get_weights()
    return model_weights


def filter_file(noisy_file, model_weights, request):
    voice_extractor = VoiceExtractor()
    voice_extractor.create_model()
    voice_extractor.model.set_weights(model_weights)
    # _, temp_file_path = tempfile.mkstemp(suffix='.wav')
    # output_path = 'files/' + os.path.basename(temp_file_path)
    # request.session['output_path'] = output_path
    output_path = 'files/output.wav'
    voice_extractor.filter_file(noisy_file, output_path)


def output(request):
    # file_path = request.session['output_path']
    file_path = 'files/output.wav'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read())
            response['Content-Disposition'] = 'inline; filename=output.wav'  # + os.path.basename(file_path)
            return response
    raise Http404
