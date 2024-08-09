#include "ETHash.hlsli"

// The cache. Each entry is 512 bits / 64 bytes
// This should be filled in before kernel execution
// as the cache cannot be computed in parallel
// cache must be at least numCacheElements long
[[vk::binding(0, 0)]]
RWStructuredBuffer<uint512> cache : register(u4);

// numCacheElements is the _element_ count of the cache.
// aka rows in some documentation

struct shader_constant {
    uint numCacheElements;
    uint numDatasetElements;
    uint datasetGenerationOffset;
};

[[vk::push_constant]]
struct shader_constant shader_constant;

void cacheLoad(out uint data[16], uint index) {
    data = (uint[16])cache[index].data;
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x + shader_constant.datasetGenerationOffset;

    if (index >= shader_constant.numDatasetElements)
        return;

    // initialize the mix
    uint mix[16]; // 512 bits
    cacheLoad(mix, index % shader_constant.numCacheElements);
    mix[0] ^= index;

    keccak_512_512(mix, mix);

    [loop]
    for (uint i = 0; i < DATASET_PARENTS; i++) {
	    const uint parentIndex = fnv(index ^ i, mix[i % 16]) % shader_constant.numCacheElements;
        // retrieve parent from cache
        uint parent[16];
        cacheLoad(parent, parentIndex);
        fnvHash(mix, parent);
    }

    keccak_512_512(mix, mix);

    // store mix into the desired dataset node
    datasetStore(index, mix);
}

