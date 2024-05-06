#include "AudioData.h"
#include "AudioRecorder.h"
#include "AudioPlayer.h"
#include <iostream>
using std::cout, std::endl;

int main(){
    AudioData audioData = AudioData();
    AudioRecorder recorder = AudioRecorder();
    recorder.record(audioData);
    AudioPlayer player = AudioPlayer();
    player.play(audioData);
    return EXIT_SUCCESS;
};