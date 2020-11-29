import scipy.io.wavfile as wavfile
from scipy.signal import resample
import librosa
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.graph_objects import Layout, Scatter, Figure
from plotly.offline import plot
from tensorflow import keras, config
from tensorflow.keras import layers, regularizers, Model, metrics, losses, Sequential
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class VoiceExtractor():
    def __init__(self, fs=16e3, win_len=256, num_segments=8*2, bits=16):
        self.fs = int(fs)
        self.bits = bits
        self.win_len = win_len
        self.num_segments = num_segments

        self.overlap = round(0.25 * win_len)  # overlap of 75%
        self.fft_len = win_len
        self.num_features = self.fft_len//2 + 1
        self.window = signal.windows.hamming(win_len, sym=False)

        self.n_samples = None
        self.train_data = None
        self.target_data = None
        self.model = None
        self.history = None
        self.noisy_stft = None

    def _load_file(self, path):  # path to a file or IO_Bytes object
        sample_rate, data = wavfile.read(path)  # 0.140625
        data = data.sum(axis=1) / 2
        data = data / (2 ** (16 - 1))
        number_of_samples = round(len(data) * float(self.fs) / sample_rate)
        data = resample(data, number_of_samples)
        return np.float32(data)

    def load_data(self, train_path, target_path):  # loads and slices data
        print('Loading data...')
        self.train_data = self._load_file(train_path)
        self.target_data = self._load_file(target_path)
        print(' Done')

    def _stft_seg(self):
        data = None
        stft = librosa.stft(self.train_data, n_fft=self.fft_len, win_length=self.win_len, hop_length=self.overlap,
                            window=self.window, center=True)
        self.noisy_stft = stft
        stft_mag = np.abs(stft)

        mean = np.mean(stft_mag)
        std = np.std(stft_mag)
        stft_mag = (stft_mag - mean) / std  # normalization

        noisy_stft = np.concatenate([stft_mag[:, 0:self.num_segments - 1], stft_mag], axis=1)
        stft_segments = np.zeros((self.num_features, self.num_segments, noisy_stft.shape[1] - self.num_segments + 1))
        for index in range(noisy_stft.shape[1] - self.num_segments + 1):
            stft_segments[:, :, index] = noisy_stft[:, index:index + self.num_segments]

        stft_segments = np.moveaxis(stft_segments, 2, 0)
        stft_segments = np.expand_dims(stft_segments, 3)

        if data is None:
            data = stft_segments
        else:
            data = np.concatenate((data, stft_segments), 0)
        return data

    def _stft_targ(self):
        data = None
        stft = librosa.stft(self.target_data, n_fft=self.fft_len, win_length=self.win_len, hop_length=self.overlap,
                            window=self.window, center=True)
        snr = np.divide(np.abs(stft), np.abs(self.noisy_stft))
        mask = np.around(snr, 0)
        mask[np.isnan(mask)] = 1
        mask[mask > 1] = 1

        mask = np.moveaxis(mask, 1, 0)
        mask = np.expand_dims(mask, 2)
        mask = np.expand_dims(mask, 3)

        if data is None:
            data = mask
        else:
            data = np.concatenate((data, mask), 0)
        return data

    def _stft_pred(self, data):
        stft = librosa.stft(data, n_fft=self.fft_len, win_length=self.win_len, hop_length=self.overlap,
                            window=self.window, center=True)
        stft_mag = np.abs(stft)

        mean = np.mean(stft_mag)
        std = np.std(stft_mag)
        stft_mag = (stft_mag - mean) / std  # normalization

        noisy_stft = np.concatenate([stft_mag[:, 0:self.num_segments - 1], stft_mag], axis=1)
        stft_segments = np.zeros((self.num_features, self.num_segments, noisy_stft.shape[1] - self.num_segments + 1))
        for index in range(noisy_stft.shape[1] - self.num_segments + 1):
            stft_segments[:, :, index] = noisy_stft[:, index:index + self.num_segments]

        stftSegments = np.moveaxis(stft_segments, 2, 0)
        stftSegments = np.expand_dims(stftSegments, 3)
        return stftSegments, stft

    def create_model(self, l2_strength=0.0):
        inputs = layers.Input(shape=[self.num_features, self.num_segments, 1])
        x = inputs

        # -----
        x = layers.ZeroPadding2D(((4, 4), (0, 0)))(x)
        x = layers.Conv2D(filters=18, kernel_size=[9, self.num_segments], strides=[1, 10], padding='valid',
                          use_bias=False, kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        skip0 = layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 10], padding='same', use_bias=False,
                       kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(skip0)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 10], padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        # -----
        x = layers.Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 10], padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        #
        skip1 = layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 10], padding='same', use_bias=False,
                       kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(skip1)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 10], padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        # ----
        x = layers.Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 100], padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 100], padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 100], padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        # ----
        x = layers.Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 100], padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 100], padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = x + skip1
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 100], padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        # ----
        x = layers.Conv2D(filters=18, kernel_size=[9, 1], strides=[1, 100], padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters=30, kernel_size=[5, 1], strides=[1, 100], padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = x + skip0
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv2D(filters=8, kernel_size=[9, 1], strides=[1, 100], padding='same', use_bias=False,
                          kernel_regularizer=regularizers.l2(l2_strength))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

        # ----
        x = layers.SpatialDropout2D(0.2)(x)
        x = layers.Conv2D(filters=1, kernel_size=[self.num_features, 1], strides=[1, 100], padding='same')(x)
        x = layers.Activation('sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=x)

    def compile_model(self, lr=3e-4):
        # loss=losses.Huber()
        loss='binary_crossentropy'
        # loss='mse'
        metric = 'accuracy'
        # metric = metrics.RootMeanSquaredError()
        self.model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=[metric])  # rmse

    def fit_model(self, steps=100, epochs=5, batch_size=None, validation_split=None):
        input_data = self._stft_seg()
        target_data = self._stft_targ()
        self.history = self.model.fit(input_data, target_data, steps_per_epoch=steps, epochs=epochs, shuffle=True,
                                        batch_size=batch_size, validation_split=validation_split)

    def save_model(self, path):
        self.model.save_weights(path)
        print("Saved model to disk")

    def load_model(self, path):
        self.model.load_weights(path)
        print("Loaded model from disk")

    def filter_file(self, input_path, output_path, input_bits=16):
        data = self._load_file(input_path)

        stft_seg, stft = self._stft_pred(data)

        results = self.model.predict(stft_seg)
        results = np.squeeze(results)
        results = results.T

        # print(np.amin(results), np.amax(results))

        stft3 = np.multiply(stft, results)

        invert = librosa.istft(stft3, win_length=self.win_len, hop_length=self.overlap, window=self.window, center=True)
        wavfile.write(output_path, self.fs, invert)

    def graphs(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=acc,
                            mode='lines+markers',
                            name='train accuracy'))
        fig.add_trace(go.Scatter(y=val_acc,
                            mode='lines+markers',
                            name='validation accuracy'))
        plocik = plot(fig, output_type='div')
        acc_plot = Plot(None, acc, 'train accuracy', 'Epochs', 'accuracy')
        return acc_plot.create()


class Plot:
    def __init__(self, x, y, title, xlabel, ylabel, ylim=None):
        self.x = x
        self.y = y
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylim = ylim

    def __repr__(self):
        return 'Formatted plot: ' + self.title

    def create(self):
        layout = Layout(paper_bgcolor='rgba(55,55,55,.4)', plot_bgcolor='rgba(36,36,36,.70)', yaxis=dict(range=self.ylim))
        data = Scatter(x=self.x, y=self.y, name="Wykres", opacity=0.1, mode='lines+markers')
        fig = Figure(data=data, layout=layout)
        fig.update_layout(
            title=self.title,
            xaxis_title=self.xlabel,
            yaxis_title=self.ylabel,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="white"
            )
        )
        fig.update_xaxes(showgrid=True, gridwidth=.3, gridcolor="rgba(100,100,100,.80)")
        fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor="rgba(100,100,100,.80)")
        return plot(fig, output_type='div')
