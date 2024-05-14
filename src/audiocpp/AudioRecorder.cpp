#include "AudioRecorder.h"
#include <iostream>

using std::cout, std::endl;

AudioRecorder::AudioRecorder() {
    // Initialize PortAudio
    PaError err;
    err = Pa_Initialize();
    AudioRecorder::checkErr(err);

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

    int device = Pa_GetDefaultInputDevice();
    inputParameters.device = device;
    const PaDeviceInfo* info = Pa_GetDeviceInfo(inputParameters.device);
    cout << "Using input device: " << info->name << endl;
    inputParameters.channelCount = 2;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.hostApiSpecificStreamInfo = nullptr;
    inputParameters.suggestedLatency = info->defaultLowInputLatency;
    double sampleRate = info->defaultSampleRate;

    err = Pa_OpenStream(
        &stream,
        &inputParameters,
        nullptr,
        sampleRate,
        paFramesPerBufferUnspecified, //TODO: change to FRAMES_PER_BUFFER?
        paNoFlag,
        myCallback,
        nullptr
    );
    checkErr(err);
}

AudioRecorder::~AudioRecorder() {
    PaError err;
    if (stream) {
        if (Pa_IsStreamActive(stream))
        err = Pa_StopStream(stream);

        err = Pa_CloseStream(stream);
        stream = nullptr;
    }

    err = Pa_Terminate();
    checkErr(err);
}

void AudioRecorder::record(AudioData& audioData) {

    audioData.isValid = false;
    audioData.samples.clear();
    PaError err;

    // Start stream and fill audiodata just once
    if (Pa_IsStreamStopped(stream)) {
        const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(inputParameters.device);
        const PaStreamInfo* streamInfo = Pa_GetStreamInfo(stream);
        audioData.device = deviceInfo->name;
        audioData.api = Pa_GetHostApiInfo(deviceInfo->hostApi)->name;
        audioData.latency = streamInfo->inputLatency;
        audioData.channels = inputParameters.channelCount;
        audioData.sampleRate = static_cast<unsigned>(streamInfo->sampleRate);
        err = Pa_StartStream(stream);
    }

    Pa_Sleep(3 * 1000);

    if (!myCallback) {
        signed long available = Pa_GetStreamReadAvailable(stream);
        cout << "Available frames: " << available << endl;
        if (available < 0) {
            checkErr(static_cast<PaError>(available));
            return;
        }
        audioData.samples.resize(available * audioData.channels);
        err = Pa_ReadStream(stream, audioData.samples.data(), available);
        checkErr(err);
        audioData.isValid = true;
    }
}


void AudioRecorder::checkErr(PaError err) {
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}

int AudioRecorder::callback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData) {

    float* in = (float*)inputBuffer;
    (void)outputBuffer;

    int dispSize = 100;
    printf("\r");
 
    float vol_l = 0;
    float vol_r = 0;

    for (unsigned long i = 0; i < framesPerBuffer * 2; i += 2) {
        vol_l = AudioRecorder::max(vol_l, std::abs(in[i]));
        vol_r = AudioRecorder::max(vol_r, std::abs(in[i+1]));
    }

    for (int i = 0; i < dispSize; i++) {
        float barProportion = i / (float)dispSize;
        if (barProportion <= vol_l && barProportion <= vol_r) {
            printf("█");
        } else if (barProportion <= vol_l) {
            printf("▀");
        } else if (barProportion <= vol_r) {
            printf("▄");
        } else {
            printf(" ");
        }
    }

    fflush(stdout);

    return 0;
}
