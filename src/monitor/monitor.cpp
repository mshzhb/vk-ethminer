#include "monitor.h"
using namespace std::chrono_literals;

void monitor::monitorHashRate() {
    bool report = false;
    std::stringstream ss;
    ss.precision(3);
    while (true)
    {
        std::this_thread::sleep_for(10000ms);
        ss.str(std::string()); //clear
        for(auto miner : miners){
            ss << "gpu" << miner->deviceIndex << " = " << miner->miningRate <<"[MH/s] ";
            report |= miner->miningRate > 0;
        }
        if(report)
            spdlog::info(prefix + "{}", ss.str());
    }
}