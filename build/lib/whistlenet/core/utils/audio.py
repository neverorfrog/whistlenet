import os
import sys
from enum import Enum

import librosa
import numpy as np
import soundfile as sf
import torch
from matplotlib import pyplot as plt

N_FFT = 1024
NUM_FREQS = (N_FFT // 2) + 1
WINDOW_SIZE = 1024
SAMPLE_RATE = 44100


class SampleType(Enum):
    No_Whistle = 0
    Whistle = 1


class Audio:
    """
    Class that contains audio data

    self.y: audio data in time series format
    self.S: audio data in frequency domain (melspectrogram) with shape (channels, freqs, frames)
    """

    def __init__(
        self,
        name=None,
        datapath=None,
        labelpath=None,
        n_fft=N_FFT,
        hop_length=N_FFT,
        sr=SAMPLE_RATE,
        y=None,
    ):
        self.name = name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.datapath = datapath
        self.labelpath = labelpath

        # converting to numpy array (timeseries)
        if y is None and name is not None:
            filepath = os.path.join(self.datapath, name) + ".wav"
            self.f = sf.SoundFile(filepath)
            self.y, _ = librosa.load(self.f, sr=self.sr, mono=False)
        else:
            self.y = y

        # frequency domain
        self.window_fn = librosa.filters.get_window("hann", Nx=n_fft)
        self.S = librosa.amplitude_to_db(
            S=np.abs(
                librosa.stft(
                    y=self.y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=self.window_fn,
                )
            ),
            ref=np.max,
        )

        self.mfcc = librosa.feature.mfcc(
            S=self.S, sr=self.sr, n_mfcc=NUM_FREQS
        )

        if self.S.ndim == 2:
            self.mfcc = np.expand_dims(self.mfcc, axis=0)
            self.S = np.expand_dims(self.S, axis=0)

        if self.S.ndim != 3:
            print(f"Expected 3 dimensions, got {self.S.ndim}")
            sys.exit(1)

        self.channels = self.S.shape[0]
        self.freqs = self.S.shape[1]
        self.frames = self.S.shape[2]

    def wave_plot(self):
        librosa.display.waveshow(self.y, sr=self.sr)

    def freq_plot(self):
        frames = librosa.time_to_frames(
            np.linspace(
                0, librosa.get_duration(y=self.y, sr=self.sr), self.S.shape[-1]
            ),
            sr=self.sr,
            hop_length=self.hop_length,
        )
        times = librosa.frames_to_time(
            frames, sr=self.sr, hop_length=self.hop_length
        )
        librosa.display.specshow(
            librosa.amplitude_to_db(self.S[0, :, :], ref=np.max),
            x_coords=frames,
            y_axis="log",
            hop_length=self.hop_length,
            ax=plt.gca(),
        )
        labels = list(zip(frames, times))
        labels = [
            f"({labels[j][0]}, \n {labels[j][1]:.2f})"
            for j in range(len(frames))
        ]
        plt.xticks(frames, labels)
        plt.show()

    def time2frame(self, time):
        return librosa.time_to_frames(
            time, sr=self.sr, hop_length=self.hop_length
        )

    def frame2time(self, frame):
        return librosa.frames_to_time(
            frame, sr=self.sr, hop_length=self.hop_length
        )

    def time2freq(self, time):
        frame = self.time2frame(time)
        return self.S[:, frame]

    def frame2freq(self, frame):
        return self.S[:, frame]

    def frame_info(self, frame):
        time = self.frame2time(frame)
        max_freq = np.argmax(self.frame2freq(frame))
        max_freq_intensity = np.max(self.frame2freq(frame))
        return time, max_freq, max_freq_intensity

    def print_info(self, start=0, end=None):
        if end is None:
            end = self.S.shape[1]
        for i in range(start, end):
            print("")
            time, max_freq, max_freq_intensity = self.frame_info(i)
            print(
                f"fram {i} \ntime {time:.2f} \nfreq {max_freq} \ninte {max_freq_intensity:.2f}"
            )

    def get_whistle_events(self):
        if self.labelpath is None:
            return
        filepath = os.path.join(self.labelpath, self.name) + ".txt"
        events = []
        with open(filepath, "r") as file:
            for line in file:
                start_time, end_time = map(float, line.strip().split())
                if end_time >= start_time:
                    events.append((start_time, end_time))
        return events

    def get_labels(self):
        """
        Returns a binary tensor of shape (len(y)/n_fft) \n
        Based on txt file in data/whistle/labels
        """
        events = self.get_whistle_events()
        labels = torch.fill(
            torch.zeros(self.channels, self.frames),
            SampleType.No_Whistle.value,
        )
        for channel in range(self.channels):
            for event in events:
                start = self.time2frame(event[0])
                end = self.time2frame(event[1])
                for i in range(start, end + 1):
                    labels[channel, i] = SampleType.Whistle.value
        return labels
