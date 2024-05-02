#include <iostream>
using std::cout, std::cin, std::endl;

#include "AudioRecorder.h"
#include "AudioPlayer.h"


int main(){
    AudioRecorder recorder = AudioRecorder();
    AudioPlayer player = AudioPlayer();
    AudioData data = AudioData();
    recorder.update(data);
    player.play(data);
}