#include "vulkanminer.h"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
    VkDebugReportFlagsEXT                       flags,
    VkDebugReportObjectTypeEXT                  objectType,
    uint64_t                                    object,
    size_t                                      location,
    int32_t                                     messageCode,
    const char* pLayerPrefix,
    const char* pMessage,
    void* pUserData) {

    printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

    return VK_FALSE;
}

std::vector<int> VulkanEthminer::listVulkanDevice(bool autoSelectGPU) {
    const std::string prefix = "[" + green + "list" + reset + "] ";
    spdlog::info(prefix + "list all devices");
    if(autoSelectGPU)
#if __arm__ || __aarch64__ //using discrete & integrate gpu for apple & android
	spdlog::info(prefix + "automatically add all discrete & integrate GPUs to execution list.");
#else // using discrete gpu for windows & linux
	spdlog::info(prefix + "automatically add all discrete GPUs to execution list.");
#endif
    //create vk instance
    createInstance();
    //In this function, we find a physical device that can be used with Vulkan.
    //So, first we will list all physical devices on the system with vkEnumeratePhysicalDevices .
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0)
        throw std::runtime_error("could not find a device with vulkan support");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    std::vector<int> discrete_gpu_list;
    for(int i = 0; i < devices.size(); i++){
        VkPhysicalDeviceProperties vkPhysicalDeviceProperties;
        vkGetPhysicalDeviceProperties(devices[i], &vkPhysicalDeviceProperties);
#if __arm__||__aarch64__ //using discrete & integrate gpu for apple & android
        if(vkPhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || vkPhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            discrete_gpu_list.push_back(i);
#else // using discrete gpu for windows & linux
        if(vkPhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            discrete_gpu_list.push_back(i);
#endif
        spdlog::info(prefix + "gpu{}: {} -> (" + magenta + "{}" + reset + ", maxBufferSize {} MB)",i, vkPhysicalDeviceProperties.deviceName,
                     vkPhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "DISCRETE_GPU" :
                     vkPhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "INTEGRATED_GPU" :
                     vkPhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? "VIRTUAL_GPU" : "OTHER_GPU",
                     vkPhysicalDeviceProperties.limits.maxStorageBufferRange >> 20);
    }
    if(autoSelectGPU)
        for(auto i : discrete_gpu_list)
            spdlog::info(prefix + red + "gpu{} added to execution list" + reset, i);
    spdlog::info("");
    return discrete_gpu_list;
}

void VulkanEthminer::initVulkanDevice() {
    // Initialize vulkan instance & device
    createInstance();
    findPhysicalDevice();
    checkVulkanFeatureSupport();
    createDevice();
    createCommandPool(commandPool);
}

void VulkanEthminer::buildDataSetGenerationPass() {
    createDescriptorPool();
    createDescriptorSetLayout();
    createDescriptorSet();
    createDatasetGenComputePipeline();
}

void VulkanEthminer::createCommandPool(VkCommandPool& commandPool)
{
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // the queue family of this command pool. All command buffers allocated from this command pool,
    // must be submitted to queue of this family ONLY.
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool))
}

void VulkanEthminer::executeDataSetGenerationPass() {
    createCommandBuffer(commandPool, commandBuffer);
    calculateDispatchSize();
    //measure time cost
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i <= dispatchInfo.dispatchRound; i++)
    {
        recordDatasetGenerationCommandBuffer(i);
        submitCommandBuffer(commandBuffer);
        //data set gen progress
        if(dispatchInfo.dispatchRound < 10 || i % (dispatchInfo.dispatchRound / 10) == 0 || i == dispatchInfo.dispatchRound) //only print of 1/10 log if dispatchRound is large
            spdlog::info(prefix + "dataset gen {} / {}", i, dispatchInfo.dispatchRound);
    }
    //measure time cost
    const auto end = std::chrono::steady_clock::now();
    //data set gen done
    spdlog::info(prefix + "dataset gen time: {}[ms]", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
#if TEST_MODE
    // Read Back Dataset and show some debug info
    for(const auto& buffer : datasetBuffers)
        readBackDataset(buffer.bufferSize, buffer.buffer, buffer.bufferMemory);
#endif
}

void VulkanEthminer::hashimotoPassThread(uint32_t thread_id)
{
    using namespace std::chrono_literals;
    while (getMinerStatus() == RUN) {
        auto& threadAsset = threadAssets[thread_id];
        recordHashimotoCommandBuffer(threadAsset.commandBuffer, threadAsset.descriptorSet, threadAsset.hashimotoNounceBase);
        submitCommandBuffer(threadAsset.commandBuffer, threadAsset.fence);
        readHashimotoSolution(threadAsset.resultBuffer, threadAsset.hashimotoNounceBase);
    }
}

void VulkanEthminer::executeHashimotoPass() {
    setMinerStatus(RUN);
    constexpr double MILLION = 1000000;
    //measure time cost
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_count; i++)
    {
        std::thread th(&VulkanEthminer::hashimotoPassThread, this, i);
        threads.push_back(std::move(th));
    }
    
    while (getMinerStatus() == RUN)
    {
        using namespace std::chrono_literals;
        const auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(10000ms);
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start; //second
        //read mining counter
        //miningCounterMutex.lock();
        miningRate = miningCounter / elapsed.count() / MILLION;
        miningCounter = 0;
        spdlog::debug(prefix + "{} mining speed = {:.2f} [MH/s]", physicalDeviceName, miningRate);
    }

    for(auto& thread : threads)
        thread.join();
}

void VulkanEthminer::prepareMiningThread(uint32_t thread_id)
{
    auto& threadAsset = threadAssets[thread_id];
    createFence(threadAsset.fence);
    createReadBackHashBuffer(threadAsset.resultBuffer);
    createDescriptorSet(threadAsset.descriptorSet, threadAsset.resultBuffer);
    createCommandPool(threadAsset.commandPool);
    createCommandBuffer(threadAsset.commandPool, threadAsset.commandBuffer);
}

void VulkanEthminer::createFence(VkFence& fence)
{
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
}

//multi-threading gpu mining pass 
void VulkanEthminer::buildHashimotoPass() {
    //create vk buffer for read back hash
    createHashimotoComputePipeline();
    threadAssets.resize(thread_count);
    for (int i = 0; i < thread_count; i++)
        prepareMiningThread(i);

    for (int i = 0; i < thread_count; i++)
        spdlog::debug(prefix + "tid[{}] resultBuffer size: {} byte", i, threadAssets[i].resultBuffer.bufferSize);
}

#ifdef TEST_MODE
void VulkanEthminer::readBackDataset(uint64_t bufferSize, VkBuffer buffer, VkDeviceMemory bufferMemory) const
{
    void* mappedMemory = nullptr;
    std::vector<uint32_t> dataset;
    dataset.resize(bufferSize / sizeof(uint32_t));
    // Map the buffer memory, so that we can read from it on the CPU.
    vkMapMemory(device, bufferMemory, 0, bufferSize, 0, &mappedMemory);
    memcpy(dataset.data(), mappedMemory, bufferSize);
    // Done reading, so unmap.
    vkUnmapMemory(device, bufferMemory);

    // Log out dataset
    spdlog::info("#dataset memVerify");
    uint64_t counter = dataset.size();
    for(const auto& data : dataset)
        counter = data == 0 ? counter - 1 : counter;

    printf("%lu / %lu (%f%%) data is valid\n", counter, dataset.size(), static_cast<double>(counter) * 100llu / dataset.size());

    for (uint32_t i = 0; i < 8; i++)
        printf("0x%08x: 0x%x\n", i * 4, dataset[i]);

    for (uint32_t i = 0x3fffffe0; i <= 0x3ffffffc; i+=4)
        printf("0x%08x: 0x%x\n", i, dataset[i / 4]);

    for (uint32_t i = dataset.size() - 8; i < dataset.size(); i++)
        printf("0x%08x: 0x%x\n", i * 4, dataset[i]);
}
#endif

void VulkanEthminer::readHashimotoSolution(struct buffer& buffer, uint64_t& hashimotoNounceBase)
{
    void* mappedMemory;
    // Map the buffer memory, so that we can read from it on the CPU.
    vkMapMemory(device, buffer.bufferMemory, 0, buffer.bufferSize, 0, &mappedMemory);
    //memcpy(hash, mappedMemory, buffer.bufferSize);
    struct hashimotoPassSolution* hashimotoPassSolution = static_cast<struct hashimotoPassSolution*>(mappedMemory);
#ifdef TEST_MODE
    for (uint32_t i = 0; i < 16; i++)
        printf("state[%x]: 0x%08x\n", i, hashimotoPassSolution->state[i]);
#else
    for(int i = 0; i < hashimotoPassSolution->count; i++){
        Stratum::solution solution = {ethash_context,
                                      hashimotoNounceBase + hashimotoPassSolution->nonce[i],
                                      miningConfig.header,
                                      miningConfig.difficulty};

        std::thread{&Stratum::submit, &stratum, solution, deviceIndex}.detach();
    }
#endif
    vkUnmapMemory(device, buffer.bufferMemory);
    //update mining counter
    miningCounter += DISPATCH_DATASET_SIZE;
    hashimotoNounceBase += DISPATCH_DATASET_SIZE;
}


void VulkanEthminer::createInstance() {
    std::vector<const char*> enabledExtensions;
    /*
    By enabling validation layers, Vulkan will emit warnings if the API
    is used incorrectly. We shall enable the layer VK_LAYER_LUNARG_standard_validation,
    which is basically a collection of several useful validation layers.
    */
    if (enableValidationLayers) {
        //We get all supported layers with vkEnumerateInstanceLayerProperties.
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> layerProperties(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());
        //And then we simply check if VK_LAYER_LUNARG_standard_validation is among the supported layers.
        bool foundLayer = false;
        for (VkLayerProperties prop : layerProperties) {
            if (strcmp("VK_LAYER_KHRONOS_validation", prop.layerName) == 0) {
                foundLayer = true;
                break;
            }
        }

        if (!foundLayer) {
            throw std::runtime_error("Layer VK_LAYER_KHRONOS_validation not supported\n");
        }
        enabledLayers.push_back("VK_LAYER_KHRONOS_validation"); // Alright, we can use this layer.

        /*
        We need to enable an extension named VK_EXT_DEBUG_REPORT_EXTENSION_NAME,
        in order to be able to print the warnings emitted by the validation layer.

        So again, we just check if the extension is among the supported extensions.
        */
        uint32_t extensionCount;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensionProperties(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionProperties.data());

        bool foundExtension = false;
        for (VkExtensionProperties prop : extensionProperties) {
            if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0) {
                foundExtension = true;
                break;
            }
        }

        if (!foundExtension)
            throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");

        enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    //Contains application info. This is actually not that important.
    //The only real important field is apiVersion.
    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "vulkan-ethminer";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "vulkan";
    applicationInfo.engineVersion = 0;
    applicationInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo = {};
    enabledExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &applicationInfo;

    // Give our desired layers and extensions to vulkan.
    createInfo.enabledLayerCount = enabledLayers.size();
    createInfo.ppEnabledLayerNames = enabledLayers.data();
    createInfo.enabledExtensionCount = enabledExtensions.size();
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    //Actually create the instance. Having created the instance, we can actually start using vulkan.
    VK_CHECK_RESULT(vkCreateInstance(
        &createInfo,
        NULL,
        &instance));

    //Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME, so that warnings emitted from the validation layer are actually printed.
    if (enableValidationLayers) {
        VkDebugReportCallbackCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        createInfo.pfnCallback = &debugReportCallbackFn;

        // We have to explicitly load this function.
        auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
        if (vkCreateDebugReportCallbackEXT == nullptr) {
            throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
        }
        // Create and register callback.
        VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &createInfo, NULL, &debugReportCallback));
    }

}

void VulkanEthminer::findPhysicalDevice() {
    //In this function, we find a physical device that can be used with Vulkan.
    //So, first we will list all physical devices on the system with vkEnumeratePhysicalDevices .
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0)
        throw std::runtime_error("could not find a device with vulkan support");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    VkPhysicalDeviceProperties vkPhysicalDeviceProperties;
    vkGetPhysicalDeviceProperties(devices[deviceIndex], &vkPhysicalDeviceProperties);

    spdlog::info(prefix + "{}", vkPhysicalDeviceProperties.deviceName);
    spdlog::debug("maxStorageBufferRange: {} MB", vkPhysicalDeviceProperties.limits.maxStorageBufferRange >> 20);
    //assert(datasetSizeLimit - 1 <= vkPhysicalDeviceProperties.limits.maxStorageBufferRange); //check 4GB buffer limit
    physicalDevice = devices[deviceIndex];
    physicalDeviceName = vkPhysicalDeviceProperties.deviceName;
    maxStorageBufferRange = vkPhysicalDeviceProperties.limits.maxStorageBufferRange;
    //config dispatch size
    configKernelSize(vkPhysicalDeviceProperties);
}

void VulkanEthminer::configKernelSize(const VkPhysicalDeviceProperties& vkPhysicalDeviceProperties){
    //automatically config kernel size
    uint32_t factor;
    if(kernel_size == 0){
        uint32_t log2maxDispatchSize = log(vkPhysicalDeviceProperties.limits.maxComputeWorkGroupCount[0]);
        factor = vkPhysicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? log2maxDispatchSize : 1.2f * (log2maxDispatchSize >> 1);
        //move kernel size between [6, 16]
        factor = std::max(factor, 6u);
        factor = std::min(factor, 16u);
    }
    //using manually specified kernel size
    else
        factor = kernel_size;
    DISPATCH_DATASET_SIZE = WORKGROUP_SIZE << factor;
    spdlog::info(prefix + green +"{} kernel size: {}" + reset, kernel_size == 0 ? "auto" : "", factor);
}

void VulkanEthminer::checkVulkanFeatureSupport() {
    //check if device support wave shuffle
    VkPhysicalDeviceSubgroupProperties subgroupProperties;
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    subgroupProperties.pNext = nullptr;

    VkPhysicalDeviceProperties2 physicalDeviceProperties;
    physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physicalDeviceProperties.pNext = &subgroupProperties;

    vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties);

    if (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT)
        spdlog::debug("VK_SUBGROUP_FEATURE_SHUFFLE supported");
    else
        spdlog::debug("VK_SUBGROUP_FEATURE_SHUFFLE not supported");
}

// Returns the index of a queue family that supports compute operations.
uint32_t VulkanEthminer::getComputeQueueFamilyIndex() {
    uint32_t queueFamilyCount;

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    // Retrieve all queue families.
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    // Now find a family that supports compute.
    uint32_t i = 0;
    for (; i < queueFamilies.size(); ++i) {
        VkQueueFamilyProperties props = queueFamilies[i];
        if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            // found a queue with compute. We're done!
            break;
        }
    }

    if (i == queueFamilies.size()) {
        throw std::runtime_error("could not find a queue family that supports operations");
    }

    return i;
}

void VulkanEthminer::createDevice() {
    //We create the logical device in this function.
    //When creating the device, we also specify what queues it has.

    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueFamilyIndex = getComputeQueueFamilyIndex(); // find queue family with compute capability.
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
    float queuePriorities = 1.0;  // we only have one queue, so this is not that important.
    queueCreateInfo.pQueuePriorities = &queuePriorities;

    /*
    Now we create the logical device. The logical device allows us to interact with the physical
    device.
    */
    VkDeviceCreateInfo deviceCreateInfo = {};

    // Specify any desired device features here. We do not need any for this application, though.
    VkPhysicalDeviceFeatures deviceFeatures = {};

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.enabledLayerCount = enabledLayers.size();  // need to specify validation layers here as well.
    deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; // when creating the logical device, we also specify what queues it has.
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
#if __APPLE__ && __ARM_ARCH  //M1 mac
    const char* extension = "VK_KHR_portability_subset";
    deviceCreateInfo.enabledExtensionCount = 1;
    deviceCreateInfo.ppEnabledExtensionNames = &extension;
#endif
    VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

    // Get a handle to the only member of the queue family.
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
}

void VulkanEthminer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

//create vk buffer for cache
void VulkanEthminer::createCacheBuffer() {

    struct buffer cacheStagingBuffer = { cacheBuffer.bufferSize, nullptr, nullptr };

    createBuffer(cacheStagingBuffer.bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        cacheStagingBuffer.buffer,
        cacheStagingBuffer.bufferMemory);

    createBuffer(cacheBuffer.bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        cacheBuffer.buffer,
        cacheBuffer.bufferMemory);

    //initialize cache staging buffer
    void* gpuData;
    vkMapMemory(device, cacheStagingBuffer.bufferMemory, 0, cacheStagingBuffer.bufferSize, 0, &gpuData);
    std::memcpy(gpuData, ethash_context->light_cache->bytes, cacheStagingBuffer.bufferSize);
    vkUnmapMemory(device, cacheStagingBuffer.bufferMemory);

    assert(cacheStagingBuffer.bufferSize == cacheBuffer.bufferSize);
    copyBuffer(cacheStagingBuffer.buffer, cacheBuffer.buffer, cacheBuffer.bufferSize);
    vkDestroyBuffer(device, cacheStagingBuffer.buffer, nullptr);
    vkFreeMemory(device, cacheStagingBuffer.bufferMemory, nullptr);
}

//create vk buffer for dataset
void VulkanEthminer::createDatasetBuffer() {
    // create buffers for datasets
    // The DAG. Note that each DAG entry is 512 bits (64byte)
    // we need to split the dataset into 2 parts
    // since we have a hard 4GB limit on the data set size (datasetSizeLimit = 0x100000000 byte)
    int64_t datasetSizeRemainder = datasetSize;
    while(datasetSizeRemainder > 0)
    {
        struct buffer datasetBuffer = {};
        datasetBuffer.bufferSize = datasetSizeRemainder > datasetSizeLimit ? datasetSizeLimit : datasetSizeRemainder;

        createBuffer(datasetBuffer.bufferSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
#if TEST_MODE
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
#else
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
#endif
                     datasetBuffer.buffer,
                     datasetBuffer.bufferMemory);

        datasetBuffers.push_back(datasetBuffer);
        datasetSizeRemainder -= datasetBuffer.bufferSize;
    }
}

//create vk buffer for read back hash
void VulkanEthminer::createReadBackHashBuffer(struct buffer& readbackBuffer) {
    //set readback buffer size
    readbackBuffer.bufferSize = sizeof(hashimotoPassSolution);
    createBuffer(readbackBuffer.bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        readbackBuffer.buffer,
        readbackBuffer.bufferMemory);
}

void VulkanEthminer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));
    VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, bufferMemory, 0));
}

// find memory type with desired properties.
uint32_t VulkanEthminer::findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    /*
    How does this search work?
    See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
    */
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memoryTypeBits & (1 << i)) &&
            ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
            return i;
    }
    return -1;
}

void VulkanEthminer::createDescriptorSetLayout() {
    //Here we specify a descriptor set layout. This allows us to bind our descriptors to
    //resources in the shader.

    /*
    Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
    0. This binds to
      layout(set = 0, binding = 0) buffer cache
      layout(set = 0, binding = 1) buffer buf0
      layout(set = 0, binding = 2) buffer buf1 *optional
    in compute shader.
    */
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBinding(1 + 4);
    //cache
    descriptorSetLayoutBinding[0].binding = 0; // binding = 0
    descriptorSetLayoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding[0].descriptorCount = 1;
    descriptorSetLayoutBinding[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    //dataset
    for(int i = 0; i < 4; i++){
        descriptorSetLayoutBinding[1 + i].binding = 1 + i; // binding = 1
        descriptorSetLayoutBinding[1 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding[1 + i].descriptorCount = 1;
        descriptorSetLayoutBinding[1 + i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = descriptorSetLayoutBinding.size();
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBinding.data();

    // Create the descriptor set layout.
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
}

void VulkanEthminer::createDescriptorPool() {
    //So we will allocate a descriptor set here.
    //But we need to first create a descriptor pool to do that.
    //Our descriptor pool can only allocate two storage buffer.
    VkDescriptorPoolSize descriptorPoolSize = {};
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 5 + thread_count * 5;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1 + thread_count;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

    // create descriptor pool.
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool));
}

void VulkanEthminer::createDescriptorSet() {
    //With the pool allocated, we can now allocate the descriptor set.
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    // allocate descriptor set.
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

    //Next, we need to connect our actual storage buffer with the descriptor.
    //We use vkUpdateDescriptorSets() to update the descriptor set.

    // Specify the buffer to bind to the descriptor.
    std::vector<VkDescriptorBufferInfo> descriptorBufferInfo(1 + 4);
    descriptorBufferInfo[0].buffer = cacheBuffer.buffer;
    descriptorBufferInfo[0].offset = 0;
    descriptorBufferInfo[0].range = cacheBuffer.bufferSize;

    for(int i = 0; i < 4; i++){
        descriptorBufferInfo[1 + i].buffer = datasetBuffers[i % datasetBuffers.size()].buffer;
        descriptorBufferInfo[1 + i].offset = 0;
        descriptorBufferInfo[1 + i].range = datasetBuffers[i % datasetBuffers.size()].bufferSize;
    }

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.descriptorCount = descriptorBufferInfo.size(); // update 2 descriptors.
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
    writeDescriptorSet.pBufferInfo = descriptorBufferInfo.data();

    // perform the update of the descriptor set.
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
}

//for mining threads
void VulkanEthminer::createDescriptorSet(VkDescriptorSet& descriptorSet, struct buffer& readbackBuffer) {
    descriptorPoolMutex.lock(); // should not access descriptor pool at the same time
    //With the pool allocated, we can now allocate the descriptor set.
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    // allocate descriptor set.
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

    //Next, we need to connect our actual storage buffer with the descriptor.
    //We use vkUpdateDescriptorSets() to update the descriptor set.

    // Specify the buffer to bind to the descriptor.
    std::vector<VkDescriptorBufferInfo> descriptorBufferInfo(1 + 4);
    descriptorBufferInfo[0].buffer = readbackBuffer.buffer;
    descriptorBufferInfo[0].offset = 0;
    descriptorBufferInfo[0].range = readbackBuffer.bufferSize;

    for(int i = 0; i < 4; i++){
        descriptorBufferInfo[1 + i].buffer = datasetBuffers[i % datasetBuffers.size()].buffer;
        descriptorBufferInfo[1 + i].offset = 0;
        descriptorBufferInfo[1 + i].range = datasetBuffers[i % datasetBuffers.size()].bufferSize;
    }

    VkWriteDescriptorSet writeDescriptorSet = {};
    writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
    writeDescriptorSet.dstBinding = 0;
    writeDescriptorSet.descriptorCount = descriptorBufferInfo.size(); // update 3 descriptors.
    writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
    writeDescriptorSet.pBufferInfo = descriptorBufferInfo.data();

    // perform the update of the descriptor set.
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, nullptr);
    descriptorPoolMutex.unlock();
}

// Read file into array of bytes, and cast to uint32_t*, then return.
// The data has been padded, so that it fits into an array uint32_t.
uint32_t* VulkanEthminer::readFile(uint32_t& length, const char* filename) {

    FILE* fp = fopen(filename, "rb");
    if (fp == nullptr) {
        printf("Could not find or open file: %s\n", filename);
    }

    // get file size.
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    long filesizepadded = static_cast<long>(ceil(filesize / 4.0)) * 4;

    // read file contents.
    char* str = new char[filesizepadded];
    fread(str, filesize, sizeof(char), fp);
    fclose(fp);

    // data padding.
    for (int i = filesize; i < filesizepadded; i++) {
        str[i] = 0;
    }

    length = filesizepadded;
    return (uint32_t*)str;
}

void VulkanEthminer::createDatasetGenComputePipeline() {
    //Create a shader module. A shader module basically just encapsulates some shader code.
    uint32_t length;

    uint32_t* code = readFile(length, "shaders/ETHashGenerateDataset.hlsl.cso");

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = code;
    createInfo.codeSize = length;

    VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, nullptr, &datasetGenComputeShaderModule));
    delete[] code;
    /*
    Now let us actually create the compute pipeline.
    A compute pipeline is very simple compared to a graphics pipeline.
    It only consists of a single stage with a compute shader.

    So first we specify the compute shader stage, and it's entry point(main).
    */
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = datasetGenComputeShaderModule;
    shaderStageCreateInfo.pName = "main";

    //setup push constant for dataset generation pass
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.size = sizeof(datasetGenPassPushConstants);
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    /*
    The pipeline layout allows the pipeline to access descriptor sets.
    So we just specify the descriptor set layout we created earlier.
    */
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;

    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &datasetGenPipelineLayout));

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = datasetGenPipelineLayout;

    //Now, we finally create the compute pipeline.
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, VK_NULL_HANDLE,
        1, &pipelineCreateInfo,
        NULL, &datasetGenPipeline));
}

void VulkanEthminer::createHashimotoComputePipeline() {
    //Create a shader module. A shader module basically just encapsulates some shader code.
    uint32_t length;

    const uint32_t* code = readFile(length, shader == shader::shared_memory ? "shaders/ETHashMine_sharedMemory.hlsl.cso" : "shaders/ETHashMine_waveShuffle.hlsl.cso");
    spdlog::info(prefix + "shader selection: " + green + "{}" + reset, shader == shader::shared_memory ? "shared memory" : "wave shuffle");

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = code;
    createInfo.codeSize = length;

    VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &hashimotoComputeShaderModule));
    delete[] code;
    /*
    Now let us actually create the compute pipeline.
    A compute pipeline is very simple compared to a graphics pipeline.
    It only consists of a single stage with a compute shader.

    So first we specify the compute shader stage, and it's entry point(main).
    */
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = hashimotoComputeShaderModule;
    shaderStageCreateInfo.pName = "main";

    //setup push constant for dataset generation pass
    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.size = sizeof(hashimotoPassPushConstants);
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    /*
    The pipeline layout allows the pipeline to access descriptor sets.
    So we just specify the descriptor set layout we created earlier.
    */
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;

    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &hashimotoPipelineLayout));

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = hashimotoPipelineLayout;

    //Now, we finally create the compute pipeline.
    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, VK_NULL_HANDLE,
        1, &pipelineCreateInfo,
        nullptr, &hashimotoPipeline));
}

void VulkanEthminer::createCommandBuffer(VkCommandPool& commandPool, VkCommandBuffer& commandBuffer) {
    //Now allocate a command buffer from the command pool.
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool; // specify the command pool to allocate from.
    // if the command buffer is primary, it can be directly submitted to queue.
    // A secondary buffer has to be called from some primary command buffer, and cannot be directly
    // submitted to a queue. To keep things simple, we use a primary command buffer.
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command buffer.
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)); // allocate command buffer.
}

void VulkanEthminer::calculateDispatchSize()
{
    // Calculate dispatch size
    const uint64_t datasetLength = datasetSize / HASH_BYTES; //#node in dataset
    //add check conditions for dispatch count
    assert(datasetSize % HASH_BYTES == 0);
    //Dispatch dataset gen round
    dispatchInfo.dispatchRound = ceil(datasetLength / DISPATCH_DATASET_SIZE);
    spdlog::info(prefix + "dispatch round: {}", dispatchInfo.dispatchRound);
}

void VulkanEthminer::recordDatasetGenerationCommandBuffer(uint32_t i)
{
    //Now we shall start recording commands into the newly allocated command buffer.
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo)); // start recording commands.

    //We need to bind a pipeline, AND a descriptor set before we dispatch.
    //The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, datasetGenPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, datasetGenPipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    //push constant command
    const datasetGenPassPushConstants constants = { static_cast<uint32_t>(cacheBuffer.bufferSize / HASH_BYTES),  //cache node count
                                                    static_cast<uint32_t>(datasetSize / HASH_BYTES), //dataset node count
                                                    i * DISPATCH_DATASET_SIZE}; // round offset
    //upload the matrix to the GPU via push constants
    vkCmdPushConstants(commandBuffer, datasetGenPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(datasetGenPassPushConstants), &constants);

    /*
    Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
    The number of workgroups is specified in the arguments.
    If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
    */
    assert(DISPATCH_DATASET_SIZE % WORKGROUP_SIZE == 0);
    vkCmdDispatch(commandBuffer,ceil(DISPATCH_DATASET_SIZE / WORKGROUP_SIZE),1,1);
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer)); // end recording commands.
}

void VulkanEthminer::recordHashimotoCommandBuffer(VkCommandBuffer& commandBuffer, VkDescriptorSet& descriptorSet, uint64_t& hashimotoNounceBase)
{
    //Now we shall start recording commands into the newly allocated command buffer.
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo)); // start recording commands.

    //We need to bind a pipeline, AND a descriptor set before we atchpatch.
    //The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, hashimotoPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, hashimotoPipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    //push constant command
    hashimotoPassPushConstants constants = {};
    std::memcpy(constants.header, miningConfig.header.word32s, sizeof(miningConfig.header));
    std::memcpy(constants.target, miningConfig.difficulty.word32s, sizeof(miningConfig.difficulty));
    constants.numDatasetElements = datasetSize / HASH_BYTES;
    //calculate nonce base for every round of dispatch
    constants.startNonce[0] = hashimotoNounceBase & 0xFFFFFFFF;
    constants.startNonce[1] = hashimotoNounceBase >> 32;
    //upload the matrix to the GPU via push constants
    vkCmdPushConstants(commandBuffer, hashimotoPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(hashimotoPassPushConstants), &constants);

    /*
    Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
    The number of workgroups is specified in the arguments.
    If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
    */
    vkCmdDispatch(commandBuffer,ceil(DISPATCH_DATASET_SIZE / WORKGROUP_SIZE),1, 1);
    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer)); // end recording commands.
}

void VulkanEthminer::submitCommandBuffer(const VkCommandBuffer& commandBuffer) {
    //Now we shall finally submit the recorded command buffer to a queue.
    VkPipelineStageFlags waitStage = { VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1; // submit a single command buffer
    submitInfo.pCommandBuffers = &commandBuffer; // the command buffer to submit.
    submitInfo.pWaitDstStageMask = &waitStage;
    //We create a fence.
    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

    //We submit the command buffer on the queue, at the same time giving a fence.
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
    /*
    The command will not have finished executing until the fence is signalled.
    So we wait here.
    We will directly after this read our buffer from the GPU,
    and we will not be sure that the command has finished executing unless we wait for the fence.
    Hence, we use a fence here.
    */
    VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device, fence, nullptr);
}

void VulkanEthminer::submitCommandBuffer(const VkCommandBuffer& commandBuffer, VkFence& fence) {
    //Now we shall finally submit the recorded command buffer to a queue.
    VkPipelineStageFlags waitStage = { VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT };
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1; // submit a single command buffer
    submitInfo.pCommandBuffers = &commandBuffer; // the command buffer to submit.
    submitInfo.pWaitDstStageMask = &waitStage;

    //reset fence before submit to queue
    vkResetFences(device, 1, &fence);
    //We submit the command buffer on the queue, at the same time giving a fence.
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));

    /*
    The command will not have finished executing until the fence is signalled.
    So we wait here.
    We will directly after this read our buffer from the GPU,
    and we will not be sure that the command has finished executing unless we wait for the fence.
    Hence, we use a fence here.
    */
    VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

}


void VulkanEthminer::cleanup() {
    //Clean up all Vulkan Resources.

    if (enableValidationLayers) {
        // destroy callback.
        const auto func = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
        if (func == nullptr) {
            throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
        }
        func(instance, debugReportCallback, nullptr);
    }

    //only list devices
    if(list_device)
    {
        vkDestroyInstance(instance, nullptr);
        return;
    }

    //vkFreeMemory(device, resultBuffer.bufferMemory, nullptr);
    //vkDestroyBuffer(device, resultBuffer.buffer, nullptr);
    vkFreeMemory(device, cacheBuffer.bufferMemory, nullptr);
    vkDestroyBuffer(device, cacheBuffer.buffer, nullptr);
    for(auto& datasetBuffer : datasetBuffers)
    {
        vkFreeMemory(device, datasetBuffer.bufferMemory, nullptr);
        vkDestroyBuffer(device, datasetBuffer.buffer, nullptr);
    }
    vkDestroyShaderModule(device, datasetGenComputeShaderModule, nullptr);
    vkDestroyShaderModule(device, hashimotoComputeShaderModule, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyPipelineLayout(device, datasetGenPipelineLayout, nullptr);
    vkDestroyPipelineLayout(device, hashimotoPipelineLayout, nullptr);
    vkDestroyPipeline(device, datasetGenPipeline, nullptr);
    vkDestroyPipeline(device, hashimotoPipeline, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}

//generate 16 MB cache from seed
void VulkanEthminer::configEpoch() {
    //set up epoch config
    int epoch = ethash::find_epoch_number(miningConfig.seed);
    spdlog::info(prefix + "initializing epoch: {}", epoch);
    ethash_context = &ethash::get_global_epoch_context(epoch);
    //init cache & dataset size
    cacheBuffer.bufferSize = ethash::get_light_cache_size(ethash_context->light_cache_num_items);
    spdlog::info(prefix + "cache size: {} MB", cacheBuffer.bufferSize >> 20);
    datasetSize = ethash::get_full_dataset_size(ethash_context->full_dataset_num_items);
    spdlog::info(prefix + "dataset size: {} MB", datasetSize >> 20);
    spdlog::info(prefix + "max buffer(ssbo) size: {} MB", datasetSizeLimit >> 20);
    //make sure the buffer size is correct
    assert(HASH_BYTES == sizeof(node));
    assert(cacheBuffer.bufferSize > 0 && datasetSize > 0);
    assert(cacheBuffer.bufferSize % HASH_BYTES == 0);
}

void VulkanEthminer::execute() {
    configEpoch();
    createCacheBuffer();
    createDatasetBuffer();
    //build pass 0: dataset generation
    buildDataSetGenerationPass();
    executeDataSetGenerationPass();
    //build pass 1: mining
    buildHashimotoPass();
    executeHashimotoPass();
    //clean vk objects
    cleanup();
}

#ifdef TEST_MODE
namespace {
    inline static ethash::hash256 to_hash256(const std::string& hex)
    {
        std::string hex_string = hex;
        if(hex_string.substr(0, 2) == "0x")
            hex_string = hex_string.substr(2); //remove 0x
        auto parse_digit = [](char d) -> int { return d <= '9' ? (d - '0') : (d - 'a' + 10); };
        ethash::hash256 hash = {};
        for (size_t i = 1; i < hex_string.size(); i += 2)
        {
            int h = parse_digit(hex_string[i - 1]);
            int l = parse_digit(hex_string[i]);
            hash.bytes[i / 2] = static_cast<uint8_t>((h << 4) | l);
        }
        return hash;
    }

    inline static std::string to_string(const ethash::hash256& hash256)
    {
        constexpr auto hex_chars = "0123456789abcdef";
        std::string str;
        str.reserve(sizeof(hash256) * 2);
        for (const auto& b : hash256.bytes)
        {
            str.push_back(hex_chars[static_cast<uint8_t>(b) >> 4]);
            str.push_back(hex_chars[static_cast<uint8_t>(b) & 0xf]);
        }
        return "0x" + str;
    }
}

void VulkanEthminer::setTestModeConfig(){
    std::string header = "0x214914e1de29ad0d910cdf31845f73ad534fb2d294e387cd22927392758cc334";
    std::string seed = "0x510e4e770828ddbf7f7b00ab00a9f6adaf81c0dc9cc85f1f8249c256942d61d9"; //epoch = 2
    std::string difficulty = "0x00ff1c01710000000000000000000000d1ff1c01710000000000000000000000";

    miningConfig.seed = to_hash256(seed);
    miningConfig.header = to_hash256(header);
    miningConfig.difficulty = to_hash256(difficulty);
}

#endif