#pragma once
#include <string>
#ifdef _WIN32
    #define uint uint32_t
#elif __APPLE__
    #define uint uint32_t
#endif

#define WORD_BYTES 4                    // bytes in word
#define WORD_BIT 32                     // bits in dword
#define HEX_BIT 4                       // bits in a hex
#define DATASET_BYTES_INIT (1 << 30)    // bytes in dataset at genesis
#define DATASET_BYTES_GROWTH (1 << 23)  // dataset growth per epoch
#define CACHE_BYTES_INIT (1 << 24)      // bytes in cache at genesis
#define CACHE_BYTES_GROWTH (1 << 17)    // cache growth per epoch
#define CACHE_MULTIPLIER 1024           // Size of the DAG relative to the cache
#define EPOCH_LENGTH 30000              // blocks per epoch
#define MIX_BYTES 128                   // width of mix
#define HASH_BYTES 64                   // hash length in bytes
#define HASH_WORD 16                    // hash length in WORD
#define HASH_LENGTH_IN_HEX 128          // hash length in hex number
#define DATASET_PARENTS 256             // number of parents of each dataset element
#define CACHE_ROUNDS 3                  // number of rounds in cache production
#define ACCESSES 64                     // number of accesses in hashimoto loop


const std::string cursor_line_start = "\r";
//# mining threads
constexpr uint32_t thread_count = 1;
constexpr uint64_t datasetSizeLimit = MAX_BUFFER_SIZE;            //2GB
constexpr uint32_t WORKGROUP_SIZE = 256;                           // Workgroup size in compute shader.


