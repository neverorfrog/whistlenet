#include <iostream>
#include "AudioPlayer.h"

const std::string deviceName = "pulse";
constexpr int channels = 1;
constexpr int sampleRate = 44100;
constexpr float latency = 0.1f;

using std::cout, std::cin, std::endl;


AudioPlayer::AudioPlayer() {
    // Initialize PortAudio
    PaError err;
    err = Pa_Initialize();
    AudioPlayer::checkErr(err);
    outputParameters.device = 9; //Pa_GetDefaultOutputDevice();
    // if (!deviceName.empty()){ // get device by name
    //     for (PaDeviceIndex i = 0; i < Pa_GetDeviceCount(); ++i){
    //       cout << "Device: " << Pa_GetDeviceInfo(i)->name << endl;
    //       if (std::string(Pa_GetDeviceInfo(i)->name) == deviceName){
    //           cout << "Using output device: " << deviceName << endl;
    //           outputParameters.device = i;
    //           break;
    //       }
    //     }
    // }

    // Initializing parameters
    const PaDeviceInfo* info = Pa_GetDeviceInfo(outputParameters.device);
    outputParameters.channelCount = std::min(static_cast<int>(channels), info->maxInputChannels);
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = info->defaultHighOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(&stream, nullptr, &outputParameters, sampleRate, paFramesPerBufferUnspecified, 0, nullptr, nullptr);
    checkErr(err);
}

AudioPlayer::~AudioPlayer()
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

void AudioPlayer::checkErr(PaError err) {
    if (err != paNoError) {
        cout << "PortAudio error: " << Pa_GetErrorText(err) << endl;
        exit(EXIT_FAILURE);
    }
}

void AudioPlayer::play(const AudioData& audioData) {
    PaError err;

    // Start input stream
    err = Pa_StartStream(stream);
    checkErr(err);

    // Write data
    const unsigned long frames = static_cast<unsigned long>(audioData.samples.size());
    signed long available = Pa_GetStreamWriteAvailable(stream);
    err = Pa_WriteStream(stream, audioData.samples.data(), std::min(frames, static_cast<unsigned long>(available)));
    Pa_Sleep(3 * 1000);

    checkErr(err);
}