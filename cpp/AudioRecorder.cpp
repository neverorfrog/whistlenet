#include <iostream>
#include "AudioRecorder.h"

const std::string deviceName = "pipewire";
constexpr int channels = 1;
constexpr int sampleRate = 44100;
constexpr float latency = 0.1f;

using std::cout, std::cin, std::endl;


AudioRecorder::AudioRecorder() {
    // Initialize PortAudio
    PaError err;
    err = Pa_Initialize();
    AudioRecorder::checkErr(err);
    inputParameters.device = 3; //Pa_GetDefaultInputDevice();
    // if (!deviceName.empty()){ // get device by name
    //     for (PaDeviceIndex i = 0; i < Pa_GetDeviceCount(); ++i){
    //       cout << "Device: " << Pa_GetDeviceInfo(i)->name << endl;
    //       if (std::string(Pa_GetDeviceInfo(i)->name) == deviceName){
    //           cout << "Using input device: " << deviceName << endl;
    //           inputParameters.device = i;
    //           // break;
    //       }
    //     }
    // }

    // Initializing parameters
    const PaDeviceInfo* info = Pa_GetDeviceInfo(inputParameters.device);
    inputParameters.channelCount = std::min(static_cast<int>(channels), info->maxInputChannels);
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = latency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(&stream, &inputParameters, nullptr, sampleRate, paFramesPerBufferUnspecified, 0, nullptr, nullptr);
    checkErr(err);
}

AudioRecorder::~AudioRecorder()
{
  if (stream)
  {
    if (Pa_IsStreamActive(stream))
      Pa_StopStream(stream);

    Pa_CloseStream(stream);
    stream = nullptr;
  }

  Pa_Terminate();
}

void AudioRecorder::checkErr(PaError err) {
    if (err != paNoError) {
        cout << "PortAudio error: " << Pa_GetErrorText(err) << endl;
        exit(EXIT_FAILURE);
    }
}

void AudioRecorder::update(AudioData& audioData) {
    PaError err;

    // Start input stream
    err = Pa_StartStream(stream);
    checkErr(err);
    Pa_Sleep(3 * 1000);

    // Read data
    signed long available = Pa_GetStreamReadAvailable(stream);
    audioData.samples.resize(available * channels);
    err = Pa_ReadStream(stream, audioData.samples.data(), available);
    checkErr(err);
}