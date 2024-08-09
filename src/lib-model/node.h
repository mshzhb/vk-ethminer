#pragma once
#include <iostream>
#define node_512t node

class node{
public:
    //element in the cache
    union{
        uint8_t  byte[64];
        uint32_t word[16];
        uint64_t dword[8];
    };

    friend std::ostream& operator<<(std::ostream &os, const node& n);
    friend node operator^(node const &ln, node const &rn);
};

class node_256t {
public:
    //element in the node_256t
    union {
        uint8_t  byte[32];
        uint32_t word[8];
        uint64_t dword[4];
    };
};

