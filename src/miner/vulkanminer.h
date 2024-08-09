#pragma once
class VulkanEthminer;
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <spdlog/spdlog.h>
#include <ethash/ethash.hpp>
#include <random>
#include "../config/config.h"
#include "../config/helper.h"
#include "../lib-model/node.h"
#include "../lib-hash/sha3.h"
#include "../stratum/stratum.h"


//The application launches a compute shader that renders the mandelbrot set,
//by rendering it into a storage buffer.
//The storage buffer is then read from the GPU, and saved as .png.
class VulkanEthminer {
private:
    //program version
    std::string version = "0.1";
    //device prefix
    std::string prefix;
    //dispatch size
    int kernel_size;
    uint32_t DISPATCH_DATASET_SIZE = WORKGROUP_SIZE << 16;  // Total number of thread processed each dispatch
    //ethash context
    const ethash::epoch_context* ethash_context;
    //pool connection
    Stratum& stratum;
    //In order to use Vulkan, you must create an instance.
    VkInstance instance;
    VkDebugReportCallbackEXT debugReportCallback;

    //The physical device is some device on the system that supports usage of Vulkan.
    //Often, it is simply a graphics card that supports Vulkan.
    VkPhysicalDevice physicalDevice;
    std::string physicalDeviceName;
    uint64_t maxStorageBufferRange; //in byte
    //Then we have the logical device VkDevice, which basically allows
    //us to interact with the physical device.
    VkDevice device;

    //The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.
    //We will be creating a simple compute pipeline in this application.
    VkPipeline datasetGenPipeline;
    VkPipeline hashimotoPipeline;
    VkPipelineLayout datasetGenPipelineLayout;
    VkPipelineLayout  hashimotoPipelineLayout;
    VkShaderModule datasetGenComputeShaderModule;
    VkShaderModule hashimotoComputeShaderModule;

    //The command buffer is used to record commands, that will be submitted to a queue.
    //To allocate such command buffers, we use a command pool.
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    //Descriptors represent resources in shaders. They allow us to use things like
    //uniform buffers, storage buffers and images in GLSL.
    //A single descriptor represents a single resource, and several descriptors are organized
    //into descriptor sets, which are basically just collections of descriptors.
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    std::vector<const char *> enabledLayers;
    //In order to execute commands on a device(GPU), the commands must be submitted
    //to a queue. The commands are stored in a command buffer, and this command buffer
    //is given to the queue.
    //There will be different kinds of queues on the device. Not all queues support
    //graphics operations, for instance. For this application, we at least want a queue
    //that supports compute operations.
    VkQueue queue; // a queue supporting compute operations.
    //Groups of queues that have the same capabilities(for instance, they all support graphics and computer operations),
    //are grouped into queue families.
    //When submitting a command buffer, you must specify to which queue in the family you are submitting to.
    //This variable keeps track of the index of that queue in its family.
    uint32_t queueFamilyIndex;

public:
    bool list_device = false;
    node_256t header;
    node seed;


    //dispatch size
    struct cmdDispatchInfo{
        uint32_t dispatchRound;
    };
    // block dispatch size + dataset size
    struct cmdDispatchInfo dispatchInfo;

    //push constant for dataset generation pass
    struct datasetGenPassPushConstants {
        uint32_t cacheLength;
        uint32_t datasetLength;
        uint32_t datasetGenerationOffset;
    };


    #define MAX_SOLUTION_FOUND 3
    struct hashimotoPassSolution {
        uint32_t  count;    // 16 bytes
        uint32_t  nonce[MAX_SOLUTION_FOUND];
#if DEBUG
        uint32_t  state[16];
#endif
    };

    //push constant for dataset generation pass
    struct hashimotoPassPushConstants {
        uint32_t startNonce[2];
        uint32_t numDatasetElements;
        uint32_t pad;
        uint32_t target[8];
        uint32_t header[8];
    };

    //The struct will hold buffers.
    struct buffer
	{
        uint64_t bufferSize; // size of `buffer` in bytes.
        VkBuffer buffer;
        VkDeviceMemory bufferMemory; //The memory that backs the buffer is bufferMemory.
    };

    //The pseudorandom cache generate from seed will be saved in this vector & buffer.
    struct buffer cacheBuffer;
    //The dataset generate from cache will be hold in this buffer.
    //4 GB per dataset --> limit from VK/GPU
    uint64_t datasetSize;
    std::vector<struct buffer> datasetBuffers;

    struct threadAsset
    {
        //The dataset generate from cache will be hold in this buffer.
        struct buffer resultBuffer;
        VkCommandPool commandPool;
        VkCommandBuffer commandBuffer;
        VkDescriptorSet descriptorSet;
        VkFence fence;
        uint64_t hashimotoNounceBase = 0;
    };
    //asset for all mining threads
    std::vector<struct threadAsset> threadAssets;

    //std::mutex vkThreadMutex; //lock for vulkan queue submit
    std::mutex miningCounterMutex;
    std::mutex descriptorPoolMutex;
    int64_t miningCounter = 0;
    double miningRate = 0;
    uint64_t deviceIndex;
    //random nonce generator
    std::random_device random_device;
    //shader selection
    enum class shader { wave_shuffle, shared_memory };
    shader shader;
    //constructor: init block_number & initial seed
    VulkanEthminer(int deviceIndex, Stratum& stratum, const std::string& shader, const int kernel_size) : deviceIndex(deviceIndex), stratum(stratum), kernel_size(kernel_size)
    {
        //set device prefix
        prefix = "[" + colors[deviceIndex % colors.size()] + "gpu" + std::to_string(deviceIndex)+ reset + "] ";
        //set gpu env var
        setEnvironmentVariable("GPU_FORCE_64BIT_PTR", "0", 1);
        setEnvironmentVariable("GPU_MAX_HEAP_SIZE", "1", 1);
        setEnvironmentVariable("GPU_USE_SYNC_OBJECTS", "1", 1);
        setEnvironmentVariable("GPU_MAX_ALLOC_PERCENT", "100", 1);
        setEnvironmentVariable("GPU_SINGLE_ALLOC_PERCENT", "100", 1);
        //shader selection
        this->shader = shader == "wave_shuffle" ? shader::wave_shuffle : shader::shared_memory;
        //init Vulkan
        initVulkanDevice();
    }
    //default constructor
    VulkanEthminer(Stratum& stratum) : stratum(stratum)
    {
        list_device = true;
    }

    ~VulkanEthminer(){
        cleanup();
    }

    void initVulkanDevice();
    void buildDataSetGenerationPass();
    void executeDataSetGenerationPass();
    void hashimotoPassThread(uint32_t thread_id);
    void executeHashimotoPass();
    void prepareMiningThread(uint32_t thread_id);
    void buildHashimotoPass();
    void readBackDataset(uint64_t bufferSize, VkBuffer buffer, VkDeviceMemory bufferMemory) const;
    inline void readHashimotoSolution(buffer& buffer, uint64_t& hashimotoNounceBase);
    void createInstance();
    void findPhysicalDevice();
    void checkVulkanFeatureSupport();
    uint32_t getComputeQueueFamilyIndex();
    void createDevice();
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void createCacheBuffer();
    void createDatasetBuffer();
    void createReadBackHashBuffer(buffer& readbackBuffer);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory);
    uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties);
    void createDescriptorSetLayout();
    void createCommandPool(VkCommandPool& commandPool);
    void createDescriptorPool();
    void createDescriptorSet();
    void createDescriptorSet(VkDescriptorSet& descriptorSet, buffer& readbackBuffer);
    uint32_t* readFile(uint32_t& length, const char* filename);
    void createDatasetGenComputePipeline();
    void createHashimotoComputePipeline();
    void createCommandBuffer(VkCommandPool& commandPool, VkCommandBuffer& commandBuffer);
    void calculateDispatchSize();
    void recordDatasetGenerationCommandBuffer(uint32_t i);
    inline void recordHashimotoCommandBuffer(VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, uint64_t& hashimotoNounceBase);
    inline void submitCommandBuffer(const VkCommandBuffer& commandBuffer);
    void cleanup();
    void configEpoch();
    void keccak512(node& output, const node& input);
    //pool functions
    enum status{ RUN, STOP };
    enum status status = STOP;
    enum status getMinerStatus() { return status; }
    void setMinerStatus(enum status newStatus) {status = newStatus;}
    struct miningConfig{
        ethash_hash256 header;
        ethash_hash256 seed;
        ethash_hash256 difficulty;
    };
    miningConfig miningConfig;
    std::mutex poolConfigLock;
    void updateSeed(ethash_hash256& seed) {
        poolConfigLock.lock();
        miningConfig.seed = seed;
        poolConfigLock.unlock();
    };

    void updateHeader(ethash_hash256& header){
        //generate random nonce
        std::mt19937 gen(random_device());
        std::uniform_int_distribution<uint64_t> dis(0, ~1ull);
        uint64_t nounce = dis(gen);
        poolConfigLock.lock();
        miningConfig.header = header;
        for(uint64_t i = 0; i < threadAssets.size(); i++)
            threadAssets[i].hashimotoNounceBase = nounce ^ (i << 48ull); //different thread, device works on different nounces
        poolConfigLock.unlock();
    };
    void updateDifficulty(ethash_hash256& difficulty) {
        poolConfigLock.lock();
        miningConfig.difficulty = difficulty;
        poolConfigLock.unlock();
    }

    bool setEnvironmentVariable(const char* name, const char* value, int override)
    {
        spdlog::debug("setenv: {} -> {}", name, value);
#if WIN32
        return SetEnvironmentVariable(name, value);
#else
        return setenv(name, value, override);
#endif
    }
    
    void execute();

    void submitCommandBuffer(VkCommandBuffer const &commandBuffer, VkFence &fence);

    void createFence(VkFence &fence);
#ifdef TEST_MODE
    void setTestModeConfig();
#endif
    std::vector<int> listVulkanDevice(bool autoSelectGPU);
    void configKernelSize(const VkPhysicalDeviceProperties& vkPhysicalDeviceProperties);
};
