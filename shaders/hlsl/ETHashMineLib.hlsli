//root parameter
struct push_constant  {
    //nonce
    uint startNonce_0;
    uint startNonce_1;
    uint numDatasetElements;
    uint pad;
    //target
    uint2 target_0;
    uint2 target_1;
    uint2 target_2;
    uint2 target_3;
    //header
    uint2 header_0;
    uint2 header_1;
    uint2 header_2;
    uint2 header_3;
}; // 20 words, 80 bytes total

[[vk::push_constant]]
struct push_constant push_constant;
//only used for debug
struct result_debug {
    uint2 data[8];
};

#define MAX_FOUND 3
struct result {
    uint  count;
    struct {
        uint tid;
    } nonces[MAX_FOUND];
#if DEBUG
    uint2  state[8]; //test mode only
#endif
};

typedef struct {
    uint4 data[4];
} uint512;

[[vk::binding(0, 0)]]
RWStructuredBuffer<struct result> mineResult : register(u4);
[[vk::binding(1, 0)]]
StructuredBuffer<uint512> dataset0 : register(t0);
[[vk::binding(2, 0)]]
StructuredBuffer<uint512> dataset1 : register(t1);
[[vk::binding(3, 0)]]
StructuredBuffer<uint512> dataset2 : register(t2);
[[vk::binding(4, 0)]]
StructuredBuffer<uint512> dataset3 : register(t3);

#define DATASET_INDEX_MASK ((1 << DATASET_SELECT_BIT) - 1)
void datasetLoad(out uint data[16], uint index) {
#if SINGLE_BUFFER_MODE==0
    uint dataset_selection = index >> DATASET_SELECT_BIT;
    index &= DATASET_INDEX_MASK;

    if (dataset_selection == 0)
        data = (uint[16])dataset0[index].data;
    else if (dataset_selection == 1)
        data = (uint[16])dataset1[index].data;
    else if (dataset_selection == 2)
        data = (uint[16])dataset2[index].data;
    else if (dataset_selection == 3)
        data = (uint[16])dataset3[index].data;
#else
    data = (uint[16])dataset0[index].data;
#endif
}

void datasetLoad_uint4(inout uint4 data, uint index, uint sub_index) {
    //cast hast128 index to hash64 index
	index = (index << 1) + (sub_index >> 2);
    sub_index = sub_index & 3;
#if SINGLE_BUFFER_MODE==0
    uint dataset_selection = index >> DATASET_SELECT_BIT;
    index &= DATASET_INDEX_MASK;


    if (dataset_selection == 0)
        data = dataset0[index].data[sub_index];
    else if (dataset_selection == 1)
        data = dataset1[index].data[sub_index];
    else if (dataset_selection == 2)
        data = dataset2[index].data[sub_index];
    else
        data = dataset3[index].data[sub_index];
#else
    data = dataset0[index].data[sub_index];
#endif
}

//map 128 byte index to 64 byte node
//128:  0 {uint8 0,    1,               2,    3   }   1 {uint8 0,    1,               2,    3   }
//64 :  0 {uint4 0, 1, 2, 3} , 1 {uint4 0, 1, 2, 3} , 2 {uint4 0, 1, 2, 3} , 3 {uint4 0, 1, 2, 3} 

void datasetLoad_uint8(inout uint4 data[2], uint index, uint sub_index) {
    //cast hast128 index to hash64 index
    index = (index << 1) + (sub_index >> 1);
    sub_index = (sub_index << 1) & 3; //sub_index * 2 % 4 ~ sub_index * 2 % 4 + 1
#if SINGLE_BUFFER_MODE==0
    uint dataset_selection = index >> DATASET_SELECT_BIT;
    index &= DATASET_INDEX_MASK;


    if (dataset_selection == 0)
    {
        data[0] = dataset0[index].data[sub_index];
        data[1] = dataset0[index].data[sub_index + 1];
    }
    else if (dataset_selection == 1)
    {
        data[0] = dataset1[index].data[sub_index];
        data[1] = dataset1[index].data[sub_index + 1];
    }
    else if (dataset_selection == 2)
    {
        data[0] = dataset2[index].data[sub_index];
        data[1] = dataset2[index].data[sub_index + 1];
    }
    else
    {
        data[0] = dataset3[index].data[sub_index];
        data[1] = dataset3[index].data[sub_index + 1];
    }
#else
    data[0] = dataset0[index].data[sub_index];
    data[1] = dataset0[index].data[sub_index + 1];
#endif
}


#define DATASET_SHARDS 4

#ifndef NUM_THREADS
#define NUM_THREADS 256
#endif

#ifndef KECCAK_ROUNDS
#define KECCAK_ROUNDS 24
#endif

#define DATASET_PARENTS 256
#define CACHE_ROUNDS 3
#define HASH_WORDS 16
#define ACCESSES 64

// NOTE: we're using uint2 as a 64bit integer in most cases
// this is little endian, i.e.
// uint2.x: lower bits
// uint2.y: higher bits

// FNV implementation
#define FNV_PRIME 0x01000193
#define fnv(v1, v2) (((v1) * FNV_PRIME) ^ (v2))

void fnvHash(inout uint mix[16], in uint data[16]) {
    uint i;
    uint4 mixCasted[4] = (uint4[4])mix;
    uint4 dataCasted[4] = (uint4[4])data;
    for (i = 0; i < 4; i++) {
        mixCasted[i] = fnv(mixCasted[i], dataCasted[i]);
    }
    mix = (uint[16])mixCasted;
}

// keccak (SHA-3) implementation
// OR more precisely Keccak-f[1600]
static const uint2 keccak_rndc[24] = {
    uint2(0x00000001, 0x00000000),
    uint2(0x00008082, 0x00000000),
    uint2(0x0000808a, 0x80000000),
    uint2(0x80008000, 0x80000000),
    uint2(0x0000808b, 0x00000000),
    uint2(0x80000001, 0x00000000),
    uint2(0x80008081, 0x80000000),
    uint2(0x00008009, 0x80000000),
    uint2(0x0000008a, 0x00000000),
    uint2(0x00000088, 0x00000000),
    uint2(0x80008009, 0x00000000),
    uint2(0x8000000a, 0x00000000),
    uint2(0x8000808b, 0x00000000),
    uint2(0x0000008b, 0x80000000),
    uint2(0x00008089, 0x80000000),
    uint2(0x00008003, 0x80000000),
    uint2(0x00008002, 0x80000000),
    uint2(0x00000080, 0x80000000),
    uint2(0x0000800a, 0x00000000),
    uint2(0x8000000a, 0x80000000),
    uint2(0x80008081, 0x80000000),
    uint2(0x00008080, 0x80000000),
    uint2(0x80000001, 0x00000000),
    uint2(0x80008008, 0x80000000),
};

static const uint keccak_rotc[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

static const uint keccak_piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

uint2 rotl64(uint2 a, uint b) {
    uint2 t = uint2(0, 0);
    if (b >= 32) {
        a = a.yx;
        b -= 32;
    }
    if (b == 0) {
        return a;
    }
    t.x = (a.x << b) | (a.y >> (32 - b));
    t.y = (a.y << b) | (a.x >> (32 - b));
    return t;
}

void keccak(inout uint2 st[25])
{
    // variables
    uint i, j, r;
    uint2 t, bc[5];

    // actual iteration
    for (r = 0; r < KECCAK_ROUNDS; r++) {
        // theta
        for (i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

        for (i = 0; i < 5; i++) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        // rho pi
        t = st[1];
        for (i = 0; i < 24; i++) {
            j = keccak_piln[i];
            bc[0] = st[j];
            st[j] = rotl64(t, keccak_rotc[i]);
            t = bc[0];
        }

        // chi
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        //  iota
        st[0] ^= keccak_rndc[r];
    }
}

// a special case for keccak 512, where input is also 512 bits
// note that this implements the original keccak, not the NIST version
void keccak_512_512(out uint dst[16], in uint src[16]) {
    uint i;
    uint2 st[25];

    for (i = 0; i < 8; i++)
        st[i] = uint2(src[i * 2], src[i * 2 + 1]);

    for (i = 8; i < 25; i++)
        st[i] = 0;

    // 64, 71
    st[8] = uint2(0x00000001, 0x80000000);

    keccak(st);

    for (i = 0; i < 8; i++) {
        dst[i * 2] = st[i].x;
        dst[i * 2 + 1] = st[i].y;
    }
}

// a special case for keccak 512, where input is 320 bits
// note that this implements the original keccak, not the NIST version
void keccak_512_320(out uint dst[16], in uint src[10]) {
    uint i;
    uint2 st[25];

    for (i = 0; i < 5; i++)
        st[i] = uint2(src[i * 2], src[i * 2 + 1]);

    for (i = 5; i < 25; i++)
        st[i] = 0;

    // 40, 71
    st[5] = uint2(0x00000001, 0x00000000);
    st[8] = uint2(0x00000000, 0x80000000);

    keccak(st);

    for (i = 0; i < 8; i++) {
        dst[i * 2] = st[i].x;
        dst[i * 2 + 1] = st[i].y;
    }
}

// a special case for keccak 512, where input is 320 bits
// note that this implements the original keccak, not the NIST version
void keccak_512_320_nh(inout uint2 dst[25], uint tidx) {
    uint i;
    uint2 st[25];

    //for (i = 0; i < 4; i++)
    //    st[i] = uint2(header[(i * 2) / 4][(i * 2) % 4], header[(i * 2 + 1) / 4][(i * 2 + 1) % 4]);
    st[0] = push_constant.header_0;
    st[1] = push_constant.header_1;
    st[2] = push_constant.header_2;
    st[3] = push_constant.header_3;

    st[4] = uint2(push_constant.startNonce_0 + tidx, push_constant.startNonce_1);
    for (i = 5; i < 25; i++)
        st[i] = 0;

    // 40, 71
    st[5] = uint2(0x00000001, 0x00000000);
    st[8] = uint2(0x00000000, 0x80000000);

    keccak(st);

    for (i = 0; i < 8; i++) {
        dst[i * 2].x = st[i].x;
        dst[i * 2 + 1].x = st[i].y;
    }
}

void keccak_256_768(out uint dst[8], in uint src[24]) {
    uint i;
    uint2 st[25];

    for (i = 0; i < 12; i++)
        st[i] = uint2(src[i * 2], src[i * 2 + 1]);

    for (i = 12; i < 25; i++)
        st[i] = 0;

    // 96 135
    st[12] = uint2(0x00000001, 0x00000000);
    st[16] = uint2(0x00000000, 0x80000000);

    keccak(st);
    for (i = 0; i < 4; i++) {
        dst[i * 2] = st[i].x;
        dst[i * 2 + 1] = st[i].y;
    }
}

uint keccak_256_768_first_entry(inout uint2 st[25]) {
    uint i;
    //uint2 st[25];

    for (i = 0; i < 12; i++)
        st[i] = uint2(st[i * 2].x, st[i * 2 + 1].x);

    for (i = 12; i < 25; i++)
        st[i] = 0;

    // 96 135
    st[12] = uint2(0x00000001, 0x00000000);
    st[16] = uint2(0x00000000, 0x80000000);

    keccak(st);
    return st[0].x;
    /*for (i = 0; i < 4; i++) {
        dst[i * 2] = st[i].x;
        dst[i * 2 + 1] = st[i].y;
    }*/
}


uint4 vectorize2(inout uint2 x, inout uint2 y) {
    uint4 result;
    result.x = x.x;
    result.y = x.y;
    result.z = y.x;
    result.w = y.y;
    return result;
}

uint4 fnv4(uint4 a, uint4 b) {
    uint4 c;
    c.x = a.x * FNV_PRIME ^ b.x;
    c.y = a.y * FNV_PRIME ^ b.y;
    c.z = a.z * FNV_PRIME ^ b.z;
    c.w = a.w * FNV_PRIME ^ b.w;
    return c;
}

uint fnv_reduce(uint4 v) {
    return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

void keccak_f1600_final(inout uint2 state[25])
{
    //prepare state
    state[12] = uint2(1, 0);
    state[13] = uint2(0, 0);
    state[14] = uint2(0, 0);
    state[15] = uint2(0, 0);
    state[16] = uint2(0, 0x80000000);
    for (uint i = 17; i < 25; i++)
        state[i] = uint2(0, 0);
    keccak(state);
}

void initHeaderNonce(inout uint2 state[25], uint3 tid)
{
    state[0] = push_constant.header_0;
    state[1] = push_constant.header_1;
    state[2] = push_constant.header_2;
    state[3] = push_constant.header_3;
    //32 bit overflow
    uint nonce_lo = push_constant.startNonce_0 + tid.x;
    uint carry = push_constant.startNonce_0 >> 31 && nonce_lo >> 31 == 0 ? 1 : 0;
    state[4] = uint2(nonce_lo, push_constant.startNonce_1 + carry);
    // 40, 71
    state[5] = uint2(1, 0);
    state[6] = uint2(0, 0);
    state[7] = uint2(0, 0);
    state[8] = uint2(0, 0x80000000);

    [unroll]
    for (uint m = 9; m < 25; m++)
        state[m] = uint2(0, 0);
}

uint byte_swap(uint input)
{
    uint re = 0;
    re |= (input & 0xff) << 24;    
    re |= (input & 0xff00) << 8;
    re |= (input & 0xff0000) >> 8;
    re |= (input & 0xff000000) >> 24;
    return re;
}
