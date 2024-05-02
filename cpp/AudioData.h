#pragma once
#include <iostream>
#include <vector>
using std::cout, std::cin, std::endl;


struct AudioData {
    bool isValid{false};
    std::vector<float> samples{};
};