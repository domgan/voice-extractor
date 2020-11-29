from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from .forms import UploadFileForm

import numpy as np
from .models import VoiceExtractor
import uuid
from django.http import HttpResponse, Http404
import os
from django.conf import settings


def home(request):
    if not os.path.exists('media'):
        os.mkdir('media')
    return render(request, 'home/index.html', {})


def train(request):
    figures = ''
    if request.method == 'POST':
        noisy_train = request.FILES['noisy_train'].file
        clear_train = request.FILES['clear_train'].file
        model_weights, figures = create_model(noisy_train, clear_train)
        model_weights = [weights.tolist() for weights in model_weights]  # change list of ndarrays to list of list for json serialization
        request.session['model_weights'] = model_weights
    # return render(request, None, {'loss': figures})
    return HttpResponse(status=204)


def filter(request):
    if request.method == 'POST':
        noisy_file = request.FILES['noisy_file'].file
        model_weights = [np.array(weights) for weights in request.session['model_weights']]
        unique_filename = filter_file(noisy_file, model_weights)
        print('Filtered file ready.')
        request.session['unique_filename'] = unique_filename
    return HttpResponse(status=204)


def download(request):
    if request.method == 'GET':
        unique_filename = request.session['unique_filename']
        file_path = os.path.join(settings.MEDIA_ROOT, unique_filename + '.wav')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                filedata = fh.read()
                response = HttpResponse(filedata, content_type='application/force-download')
                response['Content-Disposition'] = 'inline; filename=output.wav'
            os.remove(file_path)
            return response
        raise Http404


def create_model(noisy_train, clear_train):
    voice_extractor = VoiceExtractor()
    voice_extractor.load_data(noisy_train, clear_train)
    voice_extractor.create_model()
    voice_extractor.compile_model(1e-2)
    voice_extractor.fit_model(steps=1330, epochs=2, validation_split=0.05)  # 1330
    figures = voice_extractor.graphs()
    model_weights = voice_extractor.model.get_weights()
    return model_weights, figures


def filter_file(noisy_file, model_weights):
    unique_filename = str(uuid.uuid4().hex)
    file_path = 'media/{}.wav'.format(unique_filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    voice_extractor = VoiceExtractor()
    voice_extractor.create_model()
    voice_extractor.model.set_weights(model_weights)
    # _, temp_file_path = tempfile.mkstemp(suffix='.wav')
    # output_path = 'files/' + os.path.basename(temp_file_path)
    # request.session['output_path'] = output_path
    voice_extractor.filter_file(noisy_file, file_path)
    return unique_filename
