#include "AudioData.h"
#include <portaudio.h>
#include <kissfft/kiss_fft.h>

class AudioPlayer {
    private:
        PaStream* stream = nullptr;
        PaStreamParameters outputParameters;
        
    public:
        AudioPlayer();
        ~AudioPlayer();

        const PaStream* getStream() { return stream; }

        void play(const AudioData& audioData);

        /**
         * @brief Checks if a PortAudio error occurred. If so, it prints the error
         * message and exits the program.
         *
         * @param err PortAudio error code
         */
        void checkErr(PaError err);
};