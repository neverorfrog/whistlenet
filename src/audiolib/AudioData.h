#pragma once
#include "portaudio.h"
#include <string>
#include <vector>

// typedef struct
// {
//     int          frameIndex;  /* Index into sample array. */
//     int          maxFrameIndex;
//     double*      recordedSamples;
// } data;


class AudioData {
    public:
        int channels{2};
        double sampleRate{44100};
        std::string device{"default"};
        std::string api{"portaudio"};
        double latency{0.0};
        bool isValid{false};
        int framesPerBuffer{512};
        std::vector<float> samples{};
};