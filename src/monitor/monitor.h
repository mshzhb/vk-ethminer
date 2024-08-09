//
// Created by mshzhb on 1/4/23.
//
#pragma once
#include "../miner/vulkanminer.h"
class monitor {
public:
    const std::string prefix = "[" + red + "monitor" + reset + "] ";
    std::vector<VulkanEthminer*> miners;
    std::thread* monitor_thread;
    //constructor
    monitor(std::vector<VulkanEthminer*> miners) {
        this->miners = miners;
    }

    ~monitor(){
        monitor_thread->detach();
        delete monitor_thread;
    }
    //start monitor
    void execute(){
        monitor_thread = new std::thread(&monitor::monitorHashRate, this);
    }
private:
    void monitorHashRate();
};



