#include "AudioData.h"
#include <portaudio.h>


class AudioPlayer {
    private:
        void checkErr(PaError err);
        PaStream* stream = nullptr;
        PaStreamParameters outputParameters;
        
    public:
        void play(const AudioData& AudioData);
        AudioPlayer();
        ~AudioPlayer();
};