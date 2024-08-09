#include "ETHashMineLib.hlsli"
#define THREADS_PER_HASH 8
#define PARALLEL_HASH    4
#define THREADS_BASE_HASH_MASK (~(THREADS_PER_HASH - 1))



[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x == 0)
        mineResult[0].count = 0;
    // Threads work together in this phase in groups of 8.
    const uint thread_id = tid.x & (THREADS_PER_HASH - 1);
    const uint mix_idx = thread_id & 0x3u;
    //lane base id with in a wave
    const uint lane_base_id = WaveGetLaneIndex() & THREADS_BASE_HASH_MASK;

    uint2 state[25];
    initHeaderNonce(state, tid);

    //sha3_512
    keccak(state);

    for (uint i = 0; i < THREADS_PER_HASH; i += PARALLEL_HASH) {
        uint4 mix[PARALLEL_HASH];
        uint offset[PARALLEL_HASH];
        uint init0[PARALLEL_HASH];

        // share init among threads
        for (int p0 = 0; p0 < PARALLEL_HASH; p0++) {
            uint2 shuffle[8];
            [unroll]
            for (int j = 0; j < 8; j++) {
                shuffle[j].x = WaveReadLaneAt(state[j].x, i + p0 + lane_base_id);
                shuffle[j].y = WaveReadLaneAt(state[j].y, i + p0 + lane_base_id);
            }
            switch (mix_idx) {
            case 0:
                mix[p0] = vectorize2(shuffle[0], shuffle[1]);
                break;
            case 1:
                mix[p0] = vectorize2(shuffle[2], shuffle[3]);
                break;
            case 2:
                mix[p0] = vectorize2(shuffle[4], shuffle[5]);
                break;
            case 3:
                mix[p0] = vectorize2(shuffle[6], shuffle[7]);
                break;
            }
            init0[p0] = WaveReadLaneAt(shuffle[0].x, lane_base_id);
        }
     
        for (uint a = 0; a < ACCESSES; a += 4) {
            uint t = (a >> 2) & 0x7u;  // a[4:2]  bit 4-2   (a >> 2) & 0x7
            [unroll]
            for (uint b = 0; b < 4; b++) {
                [unroll]
                for (uint p1 = 0; p1 < PARALLEL_HASH; p1++) {
                    offset[p1] = fnv(init0[p1] ^ (a + b), mix[p1][b]) % (push_constant.numDatasetElements / 2);
                    offset[p1] = WaveReadLaneAt(offset[p1], t + lane_base_id);
                    uint4 data;
                    datasetLoad_uint4(data, offset[p1], thread_id);
                    mix[p1] = fnv4(mix[p1], data); //read 128 bytes = 8 * uint4 = 32 * 4 byte
                }
            }
        }
        [unroll]
        for (int p2 = 0; p2 < PARALLEL_HASH; p2++) {
            uint2 shuffle[4];
            uint thread_mix = fnv_reduce(mix[p2]);

            // update mix across threads
            shuffle[0].x = WaveReadLaneAt(thread_mix, 0 + lane_base_id);
            shuffle[0].y = WaveReadLaneAt(thread_mix, 1 + lane_base_id);
            shuffle[1].x = WaveReadLaneAt(thread_mix, 2 + lane_base_id);
            shuffle[1].y = WaveReadLaneAt(thread_mix, 3 + lane_base_id);
            shuffle[2].x = WaveReadLaneAt(thread_mix, 4 + lane_base_id);
            shuffle[2].y = WaveReadLaneAt(thread_mix, 5 + lane_base_id);
            shuffle[3].x = WaveReadLaneAt(thread_mix, 6 + lane_base_id);
            shuffle[3].y = WaveReadLaneAt(thread_mix, 7 + lane_base_id);

            if (i + p2 == thread_id) {
                // move mix into state:
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }
    keccak_f1600_final(state);

	/* debug
    if(tid.x == 0)
    {
        mineResult[0].nonces[0].tid = target_0.x;
        mineResult[0].nonces[1].tid = target_0.y;
    }
    return;
    */

    //uint final_hash = byte_swap(state[0].x);
    // Assume a 0x000000xx boundary
    //                      bit[224:255]                                      bit[192:223]
    const bool valid = ((state[0].x & 0xffffff) == 0) && ((state[0].x >> 24) <= push_constant.target_0.x); // || (state[0].x == target_0.x && state[0].y <= target_0.y);

    if (valid == false || mineResult[0].count >= MAX_FOUND)
        return;

    uint count;
    InterlockedAdd(mineResult[0].count, 1, count);
	mineResult[0].nonces[count].tid = tid.x;
}
 