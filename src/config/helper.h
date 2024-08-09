#pragma once
#include <vulkan/vulkan_core.h>
// Used for validating return values of Vulkan API calls.
// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}
//ascii title
const std::string title_ascii =  "__      __        _  _                    ______  _    _                 _                    \n"
                                 "\\ \\    / /       | || |                  |  ____|| |  | |               (_)                   \n"
                                 " \\ \\  / /  _   _ | || | __  __ _  _ __   | |__   | |_ | |__   _ __ ___   _  _ __    ___  _ __ \n"
                                 "  \\ \\/ /  | | | || || |/ / / _` || '_ \\  |  __|  | __|| '_ \\ | '_ ` _ \\ | || '_ \\  / _ \\| '__|\n"
                                 "   \\  /   | |_| || ||   < | (_| || | | | | |____ | |_ | | | || | | | | || || | | ||  __/| |   \n"
                                 "    \\/     \\__,_||_||_|\\_\\ \\__,_||_| |_| |______| \\__||_| |_||_| |_| |_||_||_| |_| \\___||_|   \n"
                                 "                                                                                             ";

//color
const std::string red("\033[0;31m");
const std::string green("\033[1;32m");
const std::string yellow("\033[1;33m");
const std::string cyan("\033[0;36m");
const std::string magenta("\033[0;35m");
const std::string reset("\033[0m");

const std::vector<std::string> colors = { magenta, yellow, green, cyan };

#if DEBUG
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif