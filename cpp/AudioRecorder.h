#include "AudioData.h"
#include <portaudio.h>
#include <kissfft/kiss_fft.h>

class AudioRecorder {
    private:
        PaStream* stream = nullptr;
        PaStreamParameters inputParameters;
        PaStreamCallback* myCallback = AudioRecorder::callback;
        
    public:
        AudioRecorder();
        ~AudioRecorder();
        void record(AudioData& audioData);

        const PaStream* getStream() { return stream; }

        /**
         * @brief Checks if a PortAudio error occurred. If so, it prints the error
         * message and exits the program.
         *
         * @param err PortAudio error code
         */
        void checkErr(PaError err);

        /**
         * @brief Guess what
         * 
         * @return float 
         */
        static float max(float a, float b) {
            return a > b ? a : b;
        };

        /**
         * @brief PortAudio callback function. This is called automatically by
         * PortAudio for each audio block.
         *
         * @param inputBuffer pointer to audio data (interleaved stereo)
         * @param outputBuffer pointer to audio data (unused)
         * @param framesPerBuffer number of frames (samples) in each buffer
         * @param timeInfo timing information (unused)
         * @param statusFlags status flags (unused)
         * @param userData pointer to AudioData object
         *
         * @return 0 if successful
         */
        static int callback(const void* inputBuffer, void* outputBuffer, unsigned long framesPerBuffer,
            const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags,void* userData);
};