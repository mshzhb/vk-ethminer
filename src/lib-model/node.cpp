#include "node.h"
#include <iomanip>
#include "../config/config.h"

std::ostream& operator<<(std::ostream& os, const node& n){
    //print whole 8 byte block
    for (const uint32_t i : n.word)
	    os << std::setfill('0') << std::setw(8) << std::hex << i;
    return os;
}

node operator^(node const &ln, node const &rn)
{
    node re = {};
    for(uint i=0; i < HASH_WORD; i++)
        re.word[i] = ln.word[i] ^ rn.word[i];
    return re;
}