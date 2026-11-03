#include "test_helpers.hpp"
#include <iostream>

bool assert(bool check, const char* msg) {
    if (!check) std::cerr << "FAIL: " << msg << '\n';
    return check;
}

