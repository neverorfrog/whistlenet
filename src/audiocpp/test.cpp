#include "AudioData.h"
#include "AudioRecorder.h"
#include "AudioPlayer.h"
#include <iostream>
using std::cout, std::endl;

int main(){
    AudioData audioData = AudioData();
    AudioRecorder recorder = AudioRecorder();
    cout << "Recording..." << endl;
    recorder.record(audioData);
    AudioPlayer player = AudioPlayer();
    cout << "Playing..." << endl;
    player.play(audioData);
    return EXIT_SUCCESS;
};