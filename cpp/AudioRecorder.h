#include "AudioData.h"

#include <portaudio.h>
// #include <kissfft/kiss_fft.h>

class AudioRecorder {
    private:
        void checkErr(PaError err);
        PaStream* stream = nullptr;
        PaStreamParameters inputParameters;
        
    public:
        void update(AudioData& AudioData);
        AudioRecorder();
        ~AudioRecorder();
};