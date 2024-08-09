#include "ETHashMineLib.hlsli"
#define WORKGROUP_SIZE 256

struct compute_hash_share
{
	uint2 data[8];
};

//shared memory
groupshared  compute_hash_share sharebuf[WORKGROUP_SIZE / 4];
groupshared  uint buffer[WORKGROUP_SIZE];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 groupThreadId  : SV_GroupThreadID)
{
	if (gid.x == 0)
		mineResult[0].count = 0;

	const uint thread_id = groupThreadId.x % 4;  //thread id in block
	const uint thread_id_lsb = thread_id & 1;
	const uint hash_id = groupThreadId.x / 4; // group index in block

	//init state
	uint2 state[25];
	initHeaderNonce(state, gid);

	//sha3_512
	keccak(state);

	uint init0;
	uint4 mix[2];
	
	for (uint tid = 0; tid < 4; tid++)
	{
		if (tid == thread_id)
		{
			sharebuf[hash_id].data[0] = state[0];
			sharebuf[hash_id].data[1] = state[1];
			sharebuf[hash_id].data[2] = state[2];
			sharebuf[hash_id].data[3] = state[3];
			sharebuf[hash_id].data[4] = state[4];
			sharebuf[hash_id].data[5] = state[5];
			sharebuf[hash_id].data[6] = state[6];
			sharebuf[hash_id].data[7] = state[7];
		}
		//sync shared memory
		GroupMemoryBarrierWithGroupSync();

		mix[0].xy = sharebuf[hash_id].data[thread_id_lsb << 2 | 0];  //uint2 0-3, uint2 4-7
		mix[0].zw = sharebuf[hash_id].data[thread_id_lsb << 2 | 1];  //uint2 0-3, uint2 4-7
		mix[1].xy = sharebuf[hash_id].data[thread_id_lsb << 2 | 2];  //uint2 0-3, uint2 4-7
		mix[1].zw = sharebuf[hash_id].data[thread_id_lsb << 2 | 3];  //uint2 0-3, uint2 4-7

		init0 = sharebuf[hash_id].data[0].x;
	
		for (uint a = 0; a < ACCESSES; a += 8)
		{
			const uint lane_idx = 4 * hash_id + a / 8 % 4;
			for (uint x = 0; x < 8; ++x)
			{
				buffer[groupThreadId.x] = fnv(init0 ^ (a | x),  mix[x >> 2][x & 0x3]) % (push_constant.numDatasetElements / 2);
				//sync shared memory
				GroupMemoryBarrierWithGroupSync();

				uint idx = buffer[lane_idx];
				uint4 data[2];
				datasetLoad_uint8(data, idx, thread_id);

				mix[0] = fnv4(mix[0], data[0]);
				mix[1] = fnv4(mix[1], data[1]);
			}
		}

		sharebuf[hash_id].data[thread_id] = uint2(fnv_reduce(mix[0]), fnv_reduce(mix[1]));

		//sync shared memory
		GroupMemoryBarrierWithGroupSync();

		if (tid == thread_id)
		{
			state[8] = sharebuf[hash_id].data[0];
			state[9] = sharebuf[hash_id].data[1];
			state[10] = sharebuf[hash_id].data[2];
			state[11] = sharebuf[hash_id].data[3];
		}

		//sync shared memory
		GroupMemoryBarrierWithGroupSync();
	}

	keccak_f1600_final(state);

#if DEBUG
    if (gid.x == 0)
        for (uint i = 0; i < 8; i++)
            mineResult[0].state[i] = state[i];
#endif

	//uint final_hash = byte_swap(state[0].x);
	// Assume a 0x000000xx boundary
	//                      bit[224:255]                                      bit[192:223]
	const bool valid = ((state[0].x & 0xffffff) == 0) && ((state[0].x >> 24) <= push_constant.target_0.x);

	if (valid == false || mineResult[0].count >= MAX_FOUND)
		return;

	uint count;
	InterlockedAdd(mineResult[0].count, 1, count);
	mineResult[0].nonces[count].tid = gid.x;
}