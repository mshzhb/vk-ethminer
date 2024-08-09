#include <cstdlib>
#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"
#include "miner/vulkanminer.h"
#include "stratum/stratum.h"
#include "monitor/monitor.h"
#include "config/console_color.h"
namespace vk_miner{
    //title
    const std::string title = "Vulkan Ethminer";
    const std::string version = "v0.1.0";
    const std::string contact = "mshzhb@gmail.com";
    //miner
    std::vector<VulkanEthminer*> miners;
    //default cli
    std::string server = "us1-etc.ethermine.org";
    std::string port = "4444";
    std::string rig = "rig";
    std::string wallet = "0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A";
    std::string shader = "wave_shuffle";
    std::vector<int> device_list;
    bool list_gpu = false;
    int kernel_size = 0; //automatically config kernel size
}

#ifdef TEST_MODE
int main() {
    Stratum stratum(vk_miner::server, vk_miner::port, vk_miner::wallet, vk_miner::rig);
    VulkanEthminer miner(1, stratum);
    miner.setTestModeConfig();
    miner.execute();
}
#else
int main(int argc, char** argv) {
    //setup console color print
    setupConsole();
    //set process priority
#ifdef WIN32
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#else
    nice(-10);
#endif
    //cli parser
    CLI::App app{vk_miner::title + " " + vk_miner::version};
    app.add_flag("-l,--list_gpu", vk_miner::list_gpu, "List all GPUs and exit.");
    app.add_option("-d,--device", vk_miner::device_list, "Devices list for the vkminer to run. e.g. -d 0 1 #run miner on gpu 0 and gpu 1");
    app.add_option("-s,--server", vk_miner::server, "Mining pool server. e.g. -s us1.ethermine.org");
    app.add_option("-p,--port", vk_miner::port, "Ming pool port. e.g. -p 4444");
    app.add_option("-r,--rig", vk_miner::rig, "Ming rig name. e.g. -r miner");
    app.add_option("-w,--wallet", vk_miner::wallet, "Wallet address. e.g. -w 0x0D405dc4889DE1512BfdeFa0007c3b6AA468E31A");
    app.add_option("-k,--kernel_size", vk_miner::kernel_size, "Provide dispatch kernel size for GPUs. Recommend using 12-18 for powerful GPUs and 4-8 for value tier. e.g. -k 15");
    app.add_option("--shader", vk_miner::shader, "Ming rig shader selection. e.g. --shader wave_shuffle / --shader shared_memory");
	CLI11_PARSE(app, argc, argv);

    //set logging level
	spdlog::set_level(spdlog::level::info);
    spdlog::info("{}", vk_miner::title);
    spdlog::info("version: {}", vk_miner::version);
    spdlog::info("contact: {}\n{}", vk_miner::contact, title_ascii);
    //stratum pool
    Stratum stratum(vk_miner::server, vk_miner::port, vk_miner::wallet, vk_miner::rig);
    //list device or automatically add all discrete GPU
    if(vk_miner::list_gpu || vk_miner::device_list.empty()){
        VulkanEthminer* miner = new VulkanEthminer(stratum);
        vk_miner::device_list = miner->listVulkanDevice(vk_miner::list_gpu == false);
        delete miner;
        if(vk_miner::list_gpu)
            return EXIT_SUCCESS;
    }
    //device list check
    if(vk_miner::device_list.empty()){
        spdlog::warn("no device added to the execution list. Minor will exit now.");
        spdlog::warn("if you wish to force use any gpu, please add -d 0");
        spdlog::warn("minor will exit now.");
        return EXIT_SUCCESS;
    }
    //setup miner
    for(auto index : vk_miner::device_list){
        VulkanEthminer* miner = new VulkanEthminer(index, stratum, vk_miner::shader, vk_miner::kernel_size);
        vk_miner::miners.push_back(miner);
    }
    //setup monitor
    monitor monitor(vk_miner::miners);
    monitor.execute();
    //setup pool connection
    stratum.attachMiners(vk_miner::miners);
    stratum.connect();
    stratum.listen();
    restoreConsole();
    return EXIT_SUCCESS;
}
#endif
