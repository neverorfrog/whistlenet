#include "AudioPlayer.h"
#include <iostream>

using std::cout, std::endl;


AudioPlayer::AudioPlayer() {
    // Initialize PortAudio
    PaError err;
    err = Pa_Initialize();
    AudioPlayer::checkErr(err);

    // Checking number of devices
    int numDevices = Pa_GetDeviceCount();
    printf("Number of devices: %d\n", numDevices);
    if (numDevices < 0) {
        printf("Error getting device count.\n");
        exit(EXIT_FAILURE);
    } else if (numDevices == 0) {
        printf("There are no available audio devices on this machine.\n");
        exit(EXIT_SUCCESS);
    }

    int device = Pa_GetDefaultOutputDevice();
    const PaDeviceInfo* info = Pa_GetDeviceInfo(outputParameters.device);
    cout << "Using output device: " << info->name << endl;
    outputParameters.device = device;
    outputParameters.channelCount = 2;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.hostApiSpecificStreamInfo = nullptr;
    outputParameters.suggestedLatency = info->defaultHighOutputLatency;
    double sampleRate = info->defaultSampleRate;

    err = Pa_OpenStream(
        &stream,
        nullptr,
        &outputParameters,
        sampleRate,
        paFramesPerBufferUnspecified, //TODO: change to FRAMES_PER_BUFFER?
        paNoFlag,
        nullptr,
        nullptr
    );
    checkErr(err);
}

AudioPlayer::~AudioPlayer() {
    PaError err;
    if (stream)
    {
        if (Pa_IsStreamActive(stream))
        err = Pa_StopStream(stream);

        err = Pa_CloseStream(stream);
        stream = nullptr;
    }

    err = Pa_Terminate();
    checkErr(err);
}


void AudioPlayer::checkErr(PaError err) {
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}

void AudioPlayer::play(const AudioData& audioData) {
    PaError err;
    if (audioData.samples.empty()) return;
    if (Pa_IsStreamStopped(stream)) {
        err = Pa_StartStream(stream);
        checkErr(err);
    }
    const unsigned long frames = static_cast<unsigned long>(audioData.samples.size() / audioData.channels);
    err = Pa_WriteStream(stream, audioData.samples.data(), audioData.samples.size());
    checkErr(err);
}
