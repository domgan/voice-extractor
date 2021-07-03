import pickle
from django.shortcuts import render
import numpy as np
from VoiceExtractor.net import VoiceExtractor
import uuid
from django.http import HttpResponse
import os
from django.conf import settings
from shutil import rmtree
from VoiceExtractor.plot import Plot

neurons = 32


def home(request):
    media_path = 'media'
    if not os.path.exists(media_path):
        os.mkdir(media_path)
    else:
        rmtree(media_path, ignore_errors=True)
        os.mkdir(media_path)
    try:
        del request.session['model_weights']
        del request.session['unique_filename']
        del request.FILES['noisy_train']
        del request.FILES['clear_train']
        del request.FILES['noisy_file']
        del request.session['history']
    except KeyError:
        pass
    return render(request, 'home/index.html', {})


def train(request):
    loaded = False
    if request.method == 'POST':
        try:
            noisy_train = request.FILES['noisy_train'].file
            clear_train = request.FILES['clear_train'].file
        except KeyError:
            print('First provide files for training!')
            noisy_train, clear_train = None, None
        if not noisy_train or not clear_train:
            try:
                modelfile = request.FILES['weights'].file
                model_weights = pickle.load(modelfile)
                history = 'Model loaded'
                loaded = True
            except KeyError:
                print('Upload model!')
                return HttpResponse(status=204)
        else:
            try:
                model_weights, history = create_model(noisy_train, clear_train)
            except ValueError:
                print('Length of files do not match!')
                return HttpResponse(status=204)
        request.session['history'] = history
        if not loaded:
            model_weights = [weights.tolist() for weights in model_weights]  # change list of ndarrays to list of list for json serialization
        request.session['model_weights'] = model_weights
    # return render(request, 'home/index.html', {'plot': plot}, status=204)
    return HttpResponse(status=204)


def filter(request):
    if request.method == 'POST':
        try:
            noisy_file = request.FILES['noisy_file'].file
        except KeyError:
            print('First upload file for filtering!')
            return HttpResponse(status=204)
        try:
            model_weights = [np.array(weights) for weights in request.session['model_weights']]
        except KeyError:
            print('Model not trained!')
            return HttpResponse(status=204)
        unique_filename = filter_file(noisy_file, model_weights)
        print('Filtered file ready.')
        request.session['unique_filename'] = unique_filename
    # return render(request, 'home/index.html', {}, status=204)
    return HttpResponse(status=204)


def download(request):
    if request.method == 'GET':
        try:
            unique_filename = request.session['unique_filename']
        except KeyError:
            print('File not filtered!')
            return HttpResponse(status=204)
        file_path = os.path.join(settings.MEDIA_ROOT, unique_filename + '.wav')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fh:
                filedata = fh.read()
                response = HttpResponse(filedata, content_type='application/force-download')
                response['Content-Disposition'] = 'inline; filename=output.wav'
            os.remove(file_path)
            return response
        print('Filter another file!')
        # raise Http404


def load(request):
    if request.method == 'POST':
        try:
            file = request.FILES['weights'].file
            with open(file, 'rb') as f:
                model_weights = pickle.load(f)
        except KeyError:
            return HttpResponse(status=204)
        return model_weights
    return HttpResponse(status=204)


def save(request):
    unique_modelname = str(uuid.uuid4().hex)
    if request.method == 'GET':
        try:
            model_weights = request.session['model_weights']
        except KeyError:
            print('First train your model!')
            return HttpResponse(status=204)
        with open(unique_modelname, 'wb') as f:
            pickle.dump(model_weights, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(unique_modelname, 'rb') as f:
            modelfile = f.read()
        response = HttpResponse(modelfile, content_type='application/force-download')
        response['Content-Disposition'] = 'inline; filename=your.model'
        os.remove(unique_modelname)
        return response


def create_model(noisy_train, clear_train):
    voice_extractor = VoiceExtractor()
    voice_extractor.load_data(noisy_train, clear_train)
    voice_extractor.create_model(neurons)
    voice_extractor.compile_model(1e-3)
    voice_extractor.fit_model(epochs=40, validation_split=0.1)  # 1330
    history = voice_extractor.history.history
    model_weights = voice_extractor.model.get_weights()
    return model_weights, history


def filter_file(noisy_file, model_weights):
    unique_filename = str(uuid.uuid4().hex)
    file_path = 'media/{}.wav'.format(unique_filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    voice_extractor = VoiceExtractor()
    voice_extractor.create_model(neurons)
    voice_extractor.model.set_weights(model_weights)
    # _, temp_file_path = tempfile.mkstemp(suffix='.wav')
    # output_path = 'files/' + os.path.basename(temp_file_path)
    # request.session['output_path'] = output_path
    voice_extractor.filter_file(noisy_file, file_path)
    return unique_filename


def plots(request):
    try:
        history = request.session['history']
    except KeyError:
        info = 'First train your model!'
        return render(request, 'home/plots.html', {'info': info})
    if type(history) is str:
        info = 'Model was loaded. No history plots!'
        return render(request, 'home/plots.html', {'info': info})
    loss = history['loss']
    val_loss = history['val_loss']
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss_plot = Plot(None, loss, 'train loss', 'Epochs', 'binary_crossentropy').create()
    val_loss_plot = Plot(None, val_loss, 'validation loss', 'Epochs', 'binary_crossentropy', color='yellow').create()
    acc_plot = Plot(None, acc, 'train accuracy', 'Epochs', 'accuracy', color='green').create()
    val_acc_plot = Plot(None, val_acc, 'validation accuracy', 'Epochs', 'accuracy', color='orange').create()

    return render(request, 'home/plots.html', {'loss_plot': loss_plot, 'val_loss_plot': val_loss_plot,
                                               'acc_plot': acc_plot, 'val_acc_plot': val_acc_plot})
