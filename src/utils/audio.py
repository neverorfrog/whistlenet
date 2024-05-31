import os

import librosa
import numpy as np
import soundfile as sf
import torch
from matplotlib import pyplot as plt


class Audio:
    """Class that contains audio data"""

    def __init__(
        self,
        name=None,
        datapath=None,
        labelpath=None,
        n_fft=1024,
        hop_length=1024,
        sr=44100,
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
            self.y, _ = librosa.load(self.f, sr=self.sr)
        else:
            self.y = y

        # short time fourier transform
        self.window_fn = librosa.filters.get_window("hamming", Nx=n_fft)
        self.S = librosa.amplitude_to_db(
            np.abs(
                librosa.stft(
                    self.y,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=self.window_fn,
                )
            )
        )

    def wave_plot(self):
        librosa.display.waveshow(self.y, sr=self.sr)

    def freq_plot(self):
        frames = librosa.time_to_frames(
            np.linspace(
                0, librosa.get_duration(y=self.y, sr=self.sr), self.S.shape[1]
            ),
            sr=self.sr,
            hop_length=self.hop_length,
        )
        times = librosa.frames_to_time(
            frames, sr=self.sr, hop_length=self.hop_length
        )
        librosa.display.specshow(
            librosa.amplitude_to_db(self.S, ref=np.max),
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
                events.append((start_time, end_time))
        return events

    def get_labels(self):
        """
        Returns a binary tensor of shape (len(y)/n_fft) \n
        Based on txt file in data/whistle/labels
        """
        events = self.get_whistle_events()
        labels = torch.zeros(self.S.shape[1])
        for event in events:
            start = self.time2frame(event[0])
            end = self.time2frame(event[1])
            for i in range(start, end + 1):
                labels[i] = 1
        return labels
