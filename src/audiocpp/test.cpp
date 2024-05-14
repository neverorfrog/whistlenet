#include "AudioData.h"
#include "AudioRecorder.h"
#include <iostream>
using std::cout, std::endl;

int main(){
    AudioData audioData = AudioData();
    AudioRecorder recorder = AudioRecorder();
    cout << "Recording..." << endl;
    recorder.record(audioData);
    return EXIT_SUCCESS;
};