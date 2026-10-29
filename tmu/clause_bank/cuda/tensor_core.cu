/*
Tensor-core clause bank.

A clause matches when the included literals are a subset of the example:
  agree = sum_k X[k] * Include[k]     (int8 0/1)
  match = agree == popcount(Include)
which is (include & ~X) == 0. SM80-SM90 uses one m16n8k256.b1 AND+POPC
instruction per eight packed words. Other supported architectures use the
m16n8k32.s8 fallback, expanding four bits into INT8 registers at load time.
The automata update is not a multiply: training freezes the include mask for
a minibatch, scores it here, then applies Type I / Type II on CUDA cores.
Predict rejects a clause with no included literals. Update does not.
n_patches > 1 is a convolutional clause: it fires when any patch matches,
and feedback reads one uniformly chosen matching patch. The host picks the
block width from the patch grid and the device limit, and sizes the patch
bitset from the device shared memory. A long literal row skips words no
clause in the panel includes when that index fits beside the bitset.
n_patches <= 1 keeps the one-row addressing.
*/
extern "C" {

extern __shared__ unsigned int tc_dyn[];

__device__ __forceinline__ unsigned tc_pack4(unsigned bits, int k0)
{
    // Four included-literal bits become four INT8 lanes of 0 or 1.
    unsigned v = (bits >> k0) & 0xfu;
    return (v & 1u)
         | ((v & 2u) << 7)
         | ((v & 4u) << 14)
         | ((v & 8u) << 21);
}

__device__ __forceinline__ void tc_mma(int &d0, int &d1, int &d2, int &d3,
                                       unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1)
{
    int o0, o1, o2, o3;
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(d0), "r"(d1), "r"(d2), "r"(d3));
    d0 = o0;
    d1 = o1;
    d2 = o2;
    d3 = o3;
}

__device__ __forceinline__ void tc_bmma_op(
    int &d0, int &d1, int &d2, int &d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ <= 900)
    int o0, o1, o2, o3;
    asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=r"(o0), "=r"(o1), "=r"(o2), "=r"(o3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(d0), "r"(d1), "r"(d2), "r"(d3));
    d0 = o0;
    d1 = o1;
    d2 = o2;
    d3 = o3;
#else
    // The runtime never selects this kernel where the PTX operation is absent.
    d0 = d1 = d2 = d3 = 0;
#endif
}

__device__ __forceinline__ unsigned tc_hash(unsigned x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ unsigned tc_xorshift(unsigned &s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

__device__ __forceinline__ float tc_u01(unsigned h)
{
    return (h >> 8) * (1.0f / 16777216.0f);
}

__device__ void tc_inc(unsigned int *ta_state, unsigned int active, int bits)
{
    unsigned int carry = active;
    for (int b = 0; b < bits; ++b) {
        unsigned int carry_next = ta_state[b] & carry;
        ta_state[b] ^= carry;
        carry = carry_next;
    }
    if (carry) {
        for (int b = 0; b < bits; ++b) {
            ta_state[b] |= carry;
        }
    }
}

__device__ void tc_dec(unsigned int *ta_state, unsigned int active, int bits)
{
    unsigned int carry = active;
    for (int b = 0; b < bits; ++b) {
        unsigned int carry_next = (~ta_state[b]) & carry;
        ta_state[b] ^= carry;
        carry = carry_next;
    }
    if (carry) {
        for (int b = 0; b < bits; ++b) {
            ta_state[b] &= ~carry;
        }
    }
}

__device__ __forceinline__ void tc_inc8(
    unsigned &p0, unsigned &p1, unsigned &p2, unsigned &p3,
    unsigned &p4, unsigned &p5, unsigned &p6, unsigned &p7,
    unsigned active)
{
    unsigned carry = active;
    unsigned next;
    next = p0 & carry; p0 ^= carry; carry = next;
    next = p1 & carry; p1 ^= carry; carry = next;
    next = p2 & carry; p2 ^= carry; carry = next;
    next = p3 & carry; p3 ^= carry; carry = next;
    next = p4 & carry; p4 ^= carry; carry = next;
    next = p5 & carry; p5 ^= carry; carry = next;
    next = p6 & carry; p6 ^= carry; carry = next;
    next = p7 & carry; p7 ^= carry; carry = next;
    if (carry) {
        p0 |= carry; p1 |= carry; p2 |= carry; p3 |= carry;
        p4 |= carry; p5 |= carry; p6 |= carry; p7 |= carry;
    }
}

__device__ __forceinline__ void tc_dec8(
    unsigned &p0, unsigned &p1, unsigned &p2, unsigned &p3,
    unsigned &p4, unsigned &p5, unsigned &p6, unsigned &p7,
    unsigned active)
{
    unsigned carry = active;
    unsigned next;
    next = (~p0) & carry; p0 ^= carry; carry = next;
    next = (~p1) & carry; p1 ^= carry; carry = next;
    next = (~p2) & carry; p2 ^= carry; carry = next;
    next = (~p3) & carry; p3 ^= carry; carry = next;
    next = (~p4) & carry; p4 ^= carry; carry = next;
    next = (~p5) & carry; p5 ^= carry; carry = next;
    next = (~p6) & carry; p6 ^= carry; carry = next;
    next = (~p7) & carry; p7 ^= carry; carry = next;
    if (carry) {
        p0 &= ~carry; p1 &= ~carry; p2 &= ~carry; p3 &= ~carry;
        p4 &= ~carry; p5 &= ~carry; p6 &= ~carry; p7 &= ~carry;
    }
}

__device__ unsigned tc_fb(unsigned seed, int cls, int clause, int sample, int chunk, float inv_s)
{
    if (inv_s >= 1.0f) {
        return 0xffffffffu;
    }
    if (inv_s <= 0.0f) {
        return 0u;
    }
    unsigned state = tc_hash(seed ^ (unsigned)(cls + 1) * 0x9e3779b9u ^
                             (unsigned)(clause + 1) * 0x85ebca6bu ^
                             (unsigned)(sample + 1) * 0xc2b2ae35u ^
                             (unsigned)(chunk + 1) * 0x27d4eb2fu);
    if (state == 0u) {
        state = 1u;
    }
    unsigned threshold = (unsigned)(inv_s * 4294967296.0f);
    unsigned mask = 0u;
    #pragma unroll 1
    for (int i = 0; i < 32; ++i) {
        if (tc_xorshift(state) < threshold) {
            mask |= 1u << i;
        }
    }
    return mask;
}

/*
One warp owns one clause of one class. Lanes split the literal words. Masks
stay bit-packed: BMMA consumes them directly and IMMA expands them in registers.
The popcount is a warp sum.
*/
__global__ void tc_extract(
    const unsigned long long *ta_ptrs,
    const unsigned int *literal_active,
    unsigned int *packed,
    int *counts,
    int *included_counts,
    int n_clauses,
    int n_slots,
    int chunks,
    int bits,
    unsigned int filter,
    int apply_literal)
{
    int lane = threadIdx.x & 31;
    int warp_id = ((int)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int total = n_slots * n_clauses;
    if (warp_id >= total) {
        return;
    }
    int cls = warp_id / n_clauses;
    int clause = warp_id - cls * n_clauses;
    int empty = ta_ptrs[cls] == 0ull;
    int pop = 0;
    int included = 0;
    const unsigned int *base = 0;
    if (!empty) {
        const unsigned int *ta = (const unsigned int *)ta_ptrs[cls];
        base = ta + ((unsigned long long)clause * (unsigned)chunks) * (unsigned)bits;
    }
    for (int k = lane; k < chunks; k += 32) {
        unsigned int m = 0u;
        if (!empty) {
            m = base[k * bits + bits - 1];
            if (k == chunks - 1) {
                m &= filter;
            }
            included += __popc(m);
            if (apply_literal) {
                m &= literal_active[k];
            }
            pop += __popc(m);
        }
        packed[(unsigned long long)warp_id * (unsigned)chunks + (unsigned)k] = m;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        pop += __shfl_xor_sync(0xffffffffu, pop, off);
        included += __shfl_xor_sync(0xffffffffu, included, off);
    }
    if (lane == 0) {
        counts[warp_id] = empty ? -1 : pop;
        included_counts[warp_id] = empty ? -1 : included;
    }
}

__global__ void tc_gather(
    const unsigned int *X,
    const int *index,
    unsigned int *dst,
    int n,
    int chunks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * chunks;
    if (i >= total) {
        return;
    }
    int s = i / chunks;
    int k = i - s * chunks;
    dst[i] = X[(unsigned long long)index[s] * (unsigned)chunks + k];
}

__device__ __forceinline__ void tc_finish_panel(
    int d0, int d1, int d2, int d3,
    const int *counts,
    const int *weights,
    unsigned char *fired,
    int *partial,
    int sample0,
    int warp,
    int g,
    int t,
    int p,
    int local,
    int cls,
    int n,
    int n_clauses,
    int n_slots,
    int reject_empty,
    int write_fired)
{
    const int *ccounts = counts + cls * n_clauses;
    const int *cw = weights + cls * n_clauses;
    unsigned char *cfired = fired;
    if (write_fired) {
        cfired += ((unsigned long long)cls * (unsigned)n_clauses) * (unsigned)n;
    }
    unsigned long long span = (unsigned long long)n * (unsigned long long)n_slots;
    int col0 = t * 2;
    int col1 = col0 + 1;
    int c0 = p * 8 + col0;
    int c1 = p * 8 + col1;
    int need0 = (c0 < n_clauses) ? __ldg(ccounts + c0) : -1;
    int need1 = (c1 < n_clauses) ? __ldg(ccounts + c1) : -1;
    if (reject_empty) {
        if (need0 == 0) need0 = -1;
        if (need1 == 0) need1 = -1;
    }
    int h0 = (c0 < n_clauses) && d0 == need0;
    int h1 = (c1 < n_clauses) && d1 == need1;
    int h2 = (c0 < n_clauses) && d2 == need0;
    int h3 = (c1 < n_clauses) && d3 == need1;
    int row0 = sample0 + warp * 16 + g;
    int row1 = row0 + 8;
    if (write_fired) {
        if (c0 < n_clauses && row0 < n) cfired[(unsigned long long)c0 * (unsigned)n + row0] = (unsigned char)h0;
        if (c1 < n_clauses && row0 < n) cfired[(unsigned long long)c1 * (unsigned)n + row0] = (unsigned char)h1;
        if (c0 < n_clauses && row1 < n) cfired[(unsigned long long)c0 * (unsigned)n + row1] = (unsigned char)h2;
        if (c1 < n_clauses && row1 < n) cfired[(unsigned long long)c1 * (unsigned)n + row1] = (unsigned char)h3;
    }
    int w0 = (c0 < n_clauses) ? __ldg(cw + c0) : 0;
    int w1 = (c1 < n_clauses) ? __ldg(cw + c1) : 0;
    int p0 = (h0 ? w0 : 0) + (h1 ? w1 : 0);
    int p1 = (h2 ? w0 : 0) + (h3 ? w1 : 0);
    p0 += __shfl_xor_sync(0xffffffff, p0, 1);
    p0 += __shfl_xor_sync(0xffffffff, p0, 2);
    p1 += __shfl_xor_sync(0xffffffff, p1, 1);
    p1 += __shfl_xor_sync(0xffffffff, p1, 2);
    if (t == 0) {
        if (row0 < n) {
            partial[(unsigned long long)local * span + (unsigned long long)row0 * (unsigned)n_slots + cls] = p0;
        }
        if (row1 < n) {
            partial[(unsigned long long)local * span + (unsigned long long)row1 * (unsigned)n_slots + cls] = p1;
        }
    }
}

/*
One block owns one row tile, one class, and one panel of 8 clauses.
grid: (row tiles, classes, panels in this group).

Eight clauses is one m16n8 tile and sixteen rows is one warp. Class scores
are the sum of panel scores. tc_reduce combines panel groups so the partial
matrix remains under the runtime memory budget.
*/
__global__ void tc_imma(
    const unsigned int *X,
    const unsigned int *masks,
    const int *counts,
    const int *weights,
    unsigned char *fired,
    int *partial,
    int n,
    int n_clauses,
    int n_slots,
    int chunks,
    int reject_empty,
    int write_fired,
    int panel_base)
{
    int tile_rows = blockDim.x >> 1;
    int sample0 = blockIdx.x * tile_rows;
    int cls = blockIdx.y;
    int local = blockIdx.z;
    int p = panel_base + local;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    int row0 = sample0 + warp * 16 + g;
    int row1 = row0 + 8;
    int bclause = p * 8 + g;

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    #pragma unroll 1
    for (int word = 0; word < chunks; ++word) {
        unsigned abit0 = row0 < n
            ? __ldg(X + (unsigned long long)row0 * (unsigned)chunks + word) : 0u;
        unsigned abit1 = row1 < n
            ? __ldg(X + (unsigned long long)row1 * (unsigned)chunks + word) : 0u;
        unsigned a0 = tc_pack4(abit0, 4 * t);
        unsigned a1 = tc_pack4(abit1, 4 * t);
        unsigned a2 = tc_pack4(abit0, 16 + 4 * t);
        unsigned a3 = tc_pack4(abit1, 16 + 4 * t);
        unsigned b0 = 0u;
        unsigned b1 = 0u;
        if (bclause < n_clauses) {
            const unsigned int *bp = masks
                + (((unsigned long long)cls * (unsigned)n_clauses + (unsigned)bclause)
                   * (unsigned)chunks + (unsigned)word);
            unsigned mask = __ldg(bp);
            b0 = tc_pack4(mask, 4 * t);
            b1 = tc_pack4(mask, 16 + 4 * t);
        }
        tc_mma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }
    tc_finish_panel(
        d0, d1, d2, d3, counts, weights, fired, partial,
        sample0, warp, g, t, p, local, cls, n, n_clauses, n_slots,
        reject_empty, write_fired);
}

/*
The binary path consumes eight packed uint32 words (K=256) per instruction.
Missing words in the final K tile and missing rows/clauses are explicit zeros.
*/
__global__ void tc_bmma(
    const unsigned int *X,
    const unsigned int *masks,
    const int *counts,
    const int *weights,
    unsigned char *fired,
    int *partial,
    int n,
    int n_clauses,
    int n_slots,
    int chunks,
    int reject_empty,
    int write_fired,
    int panel_base)
{
    int tile_rows = blockDim.x >> 1;
    int sample0 = blockIdx.x * tile_rows;
    int cls = blockIdx.y;
    int local = blockIdx.z;
    int p = panel_base + local;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    int row0 = sample0 + warp * 16 + g;
    int row1 = row0 + 8;
    int bclause = p * 8 + g;

    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    #pragma unroll 1
    for (int tile = 0; tile < chunks; tile += 8) {
        int word0 = tile + t;
        int word1 = tile + 4 + t;
        unsigned a0 = (row0 < n && word0 < chunks)
            ? __ldg(X + (unsigned long long)row0 * (unsigned)chunks + word0) : 0u;
        unsigned a1 = (row1 < n && word0 < chunks)
            ? __ldg(X + (unsigned long long)row1 * (unsigned)chunks + word0) : 0u;
        unsigned a2 = (row0 < n && word1 < chunks)
            ? __ldg(X + (unsigned long long)row0 * (unsigned)chunks + word1) : 0u;
        unsigned a3 = (row1 < n && word1 < chunks)
            ? __ldg(X + (unsigned long long)row1 * (unsigned)chunks + word1) : 0u;
        unsigned b0 = 0u;
        unsigned b1 = 0u;
        if (bclause < n_clauses) {
            const unsigned int *bp = masks
                + ((unsigned long long)cls * (unsigned)n_clauses + (unsigned)bclause)
                  * (unsigned)chunks;
            if (word0 < chunks) b0 = __ldg(bp + word0);
            if (word1 < chunks) b1 = __ldg(bp + word1);
        }
        tc_bmma_op(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
    }
    tc_finish_panel(
        d0, d1, d2, d3, counts, weights, fired, partial,
        sample0, warp, g, t, p, local, cls, n, n_clauses, n_slots,
        reject_empty, write_fired);
}

__global__ void tc_reduce(
    const int *partial,
    int *sums,
    int n,
    int n_slots,
    int panels,
    int accumulate)
{
    unsigned long long span = (unsigned long long)n * (unsigned long long)n_slots;
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= span) {
        return;
    }
    int acc = 0;
    for (int p = 0; p < panels; ++p) {
        acc += partial[(unsigned long long)p * span + idx];
    }
    if (accumulate) {
        sums[idx] += acc;
    } else {
        sums[idx] = acc;
    }
}

__device__ __forceinline__ unsigned tc_example_word(
    const unsigned int *X,
    int sample,
    int patch,
    int chunk,
    int chunks,
    int n_patches)
{
    if (n_patches <= 1) {
        return X[(unsigned long long)sample * (unsigned)chunks + (unsigned)chunk];
    }
    return X[(((unsigned long long)sample * (unsigned)n_patches) + (unsigned)patch) * (unsigned)chunks + (unsigned)chunk];
}

/*
One block owns one clause of one class. Its warps stride over literal words,
while all warps reuse one ordered byte of feedback metadata per sample.
*/
__global__ void tc_feedback(
    const unsigned long long *ta_ptrs,
    int *weights,
    const unsigned int *X,
    const unsigned char *fired_all,
    const int *sums,
    const int *targets,
    const int *not_targets,
    const int *clause_active,
    const unsigned int *literal_active,
    const int *counts,
    int n,
    int n_clauses,
    int n_slots,
    int chunks,
    int bits,
    int T,
    float inv_s,
    int boost,
    int max_included,
    int weighted,
    float type_i_p,
    float type_ii_p,
    int do_type_i,
    int do_type_ii,
    int confidence_driven,
    unsigned int filter,
    unsigned int seed,
    const int *patch_index,
    int n_patches)
{
    int tx = threadIdx.x;
    int lane = tx & 31;
    int warp = tx >> 5;
    int warps = blockDim.x >> 5;
    int clause = (int)blockIdx.x;
    int cls = (int)blockIdx.y;
    if (clause >= n_clauses || cls >= n_slots || ta_ptrs[cls] == 0ull) {
        return;
    }
    if (!clause_active[cls * n_clauses + clause]) {
        return;
    }
    unsigned char *meta = reinterpret_cast<unsigned char *>(tc_dyn);
    int aligned_n = (n + 3) & ~3;
    int *alive = reinterpret_cast<int *>(meta + aligned_n);
    const unsigned char *fired = fired_all + ((unsigned long long)cls * (unsigned)n_clauses + clause) * (unsigned)n;
    int half = n_clauses >> 1;
    int positive = clause < half;
    int includes = counts[cls * n_clauses + clause];

    if (tx == 0) {
        *alive = 0;
    }
    __syncthreads();
    for (int s = tx; s < n; s += blockDim.x) {
        int tgt = targets[s];
        int neg = not_targets[s];
        int role = 0;
        if (cls == tgt) role = 1;
        else if (cls == neg) role = 2;
        unsigned packed = 0u;
        if (role) {
            int cs = sums[(unsigned long long)s * (unsigned)n_slots + cls];
            if (cs > T) cs = T;
            if (cs < -T) cs = -T;
            float up = 0.f;
            if (T != 0) {
                up = confidence_driven
                    ? (float)(T - (cs < 0 ? -cs : cs)) / (float)T
                    : (role == 1
                        ? (float)(T - cs) / (float)(2 * T)
                        : (float)(T + cs) / (float)(2 * T));
            }
            int fired_s = fired[s] ? 1 : 0;
            int type_i = role == 1 ? positive : !positive;
            float gate = up * (type_i ? type_i_p : type_ii_p);
            unsigned h = tc_hash(seed ^ (unsigned)(cls + 1) * 0x9e3779b9u ^
                                 (unsigned)(clause + 3) * 0x85ebca6bu ^
                                 (unsigned)(s + 1) * 0xc2b2ae35u);
            int pass = gate >= 1.f || tc_u01(h) < gate;
            int do_i = type_i && do_type_i && pass;
            int do_ii = !type_i && do_type_ii && pass;
            // Nonfiring Type II feedback cannot update TA state.
            if (do_i || (do_ii && fired_s)) {
                packed = 1u | (do_i ? 2u : 0u) | (fired_s ? 4u : 0u);
                atomicExch(alive, 1);
            }
        }
        meta[s] = (unsigned char)packed;
    }
    __syncthreads();

    if (tx == 0 && weighted) {
        int w = weights[cls * n_clauses + clause];
        for (int s = 0; s < n; ++s) {
            int tgt = targets[s];
            int neg = not_targets[s];
            int role = cls == tgt ? 1 : (cls == neg ? 2 : 0);
            if (!role || !fired[s]) {
                continue;
            }
            int cs = sums[(unsigned long long)s * (unsigned)n_slots + cls];
            if (cs > T) cs = T;
            if (cs < -T) cs = -T;
            float up = 0.f;
            if (T != 0) {
                up = confidence_driven
                    ? (float)(T - (cs < 0 ? -cs : cs)) / (float)T
                    : (role == 1
                        ? (float)(T - cs) / (float)(2 * T)
                        : (float)(T + cs) / (float)(2 * T));
            }
            unsigned h = tc_hash(seed ^ 0xB5297A4Du ^ (unsigned)(cls + 1) * 0x9e3779b9u ^
                                 (unsigned)(clause + 1) * 0x85ebca6bu ^
                                 (unsigned)(s + 1) * 0xc2b2ae35u);
            if (up >= 1.f || tc_u01(h) < up) {
                if (role == 1) {
                    if (w != -1) ++w;
                } else if (w != 1) {
                    --w;
                }
            }
        }
        weights[cls * n_clauses + clause] = w;
    }

    if (*alive == 0) {
        return;
    }
    unsigned int *ta = (unsigned int *)ta_ptrs[cls];
    for (int chunk = warp * 32 + lane; chunk < chunks; chunk += warps * 32) {
        unsigned int lit = literal_active[chunk];
        if (chunk == chunks - 1) {
            lit &= filter;
        }
        if (lit == 0u) {
            continue;
        }
        unsigned int *base = ta + ((unsigned long long)clause * (unsigned)chunks + (unsigned)chunk) * (unsigned)bits;
        if (bits == 8) {
            uint4 lo = *reinterpret_cast<uint4 *>(base);
            uint4 hi = *reinterpret_cast<uint4 *>(base + 4);
            unsigned p0 = lo.x, p1 = lo.y, p2 = lo.z, p3 = lo.w;
            unsigned p4 = hi.x, p5 = hi.y, p6 = hi.z, p7 = hi.w;
            for (int s = 0; s < n; ++s) {
                unsigned packed = meta[s];
                if ((packed & 1u) == 0u) continue;
                int fired_s = (packed & 4u) != 0u;
                if (packed & 2u) {
                    if (fired_s && includes <= max_included) {
                        unsigned x = tc_example_word(
                    X, s,
                    n_patches <= 1 ? 0 : patch_index[((unsigned long long)(cls * n_clauses + clause) * (unsigned)n) + (unsigned)s],
                    chunk, chunks, n_patches);
                        if (boost) {
                            unsigned false_literals = lit & ~x;
                            tc_inc8(p0, p1, p2, p3, p4, p5, p6, p7, lit & x);
                            if (false_literals) {
                                unsigned fb = tc_fb(seed, cls, clause, s, chunk, inv_s);
                                tc_dec8(p0, p1, p2, p3, p4, p5, p6, p7, false_literals & fb);
                            }
                        } else {
                            unsigned fb = tc_fb(seed, cls, clause, s, chunk, inv_s);
                            tc_inc8(p0, p1, p2, p3, p4, p5, p6, p7, lit & x & ~fb);
                            tc_dec8(p0, p1, p2, p3, p4, p5, p6, p7, lit & ~x & fb);
                        }
                    } else {
                        unsigned fb = tc_fb(seed, cls, clause, s, chunk, inv_s);
                        tc_dec8(p0, p1, p2, p3, p4, p5, p6, p7, lit & fb);
                    }
                } else {
                    unsigned x = tc_example_word(
                    X, s,
                    n_patches <= 1 ? 0 : patch_index[((unsigned long long)(cls * n_clauses + clause) * (unsigned)n) + (unsigned)s],
                    chunk, chunks, n_patches);
                    tc_inc8(p0, p1, p2, p3, p4, p5, p6, p7, lit & ~x);
                }
            }
            lo = make_uint4(p0, p1, p2, p3);
            hi = make_uint4(p4, p5, p6, p7);
            *reinterpret_cast<uint4 *>(base) = lo;
            *reinterpret_cast<uint4 *>(base + 4) = hi;
        } else {
            unsigned int plane[16];
            for (int b = 0; b < bits; ++b) plane[b] = base[b];
            for (int s = 0; s < n; ++s) {
                unsigned packed = meta[s];
                if ((packed & 1u) == 0u) continue;
                int fired_s = (packed & 4u) != 0u;
                if (packed & 2u) {
                    if (fired_s && includes <= max_included) {
                        unsigned x = tc_example_word(
                    X, s,
                    n_patches <= 1 ? 0 : patch_index[((unsigned long long)(cls * n_clauses + clause) * (unsigned)n) + (unsigned)s],
                    chunk, chunks, n_patches);
                        if (boost) {
                            unsigned false_literals = lit & ~x;
                            tc_inc(plane, lit & x, bits);
                            if (false_literals) {
                                unsigned fb = tc_fb(seed, cls, clause, s, chunk, inv_s);
                                tc_dec(plane, false_literals & fb, bits);
                            }
                        } else {
                            unsigned fb = tc_fb(seed, cls, clause, s, chunk, inv_s);
                            tc_inc(plane, lit & x & ~fb, bits);
                            tc_dec(plane, lit & ~x & fb, bits);
                        }
                    } else {
                        unsigned fb = tc_fb(seed, cls, clause, s, chunk, inv_s);
                        tc_dec(plane, lit & fb, bits);
                    }
                } else {
                    unsigned x = tc_example_word(
                    X, s,
                    n_patches <= 1 ? 0 : patch_index[((unsigned long long)(cls * n_clauses + clause) * (unsigned)n) + (unsigned)s],
                    chunk, chunks, n_patches);
                    tc_inc(plane, lit & ~x, bits);
                }
            }
            for (int b = 0; b < bits; ++b) base[b] = plane[b];
        }
    }
}

__device__ int tc_block_sum(int value)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, off);
    }
    __shared__ int warp_sums[32];
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        int total = (lane < warps) ? warp_sums[lane] : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            total += __shfl_xor_sync(0xffffffffu, total, off);
        }
        if (lane == 0) {
            warp_sums[0] = total;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

/*
OR patch matches into one clause output. A firing clause keeps one of its
matching patches, chosen uniformly, for the frozen minibatch.
patch_fired and the outputs are clause-major: index id * n + sample.
*/
__global__ void tc_or_patches(
    const unsigned char *patch_fired,
    const int *weights,
    unsigned char *fired,
    int *patch_index,
    int *sums,
    int n,
    int n_clauses,
    int n_slots,
    int n_patches,
    unsigned seed)
{
    int clause = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int sample = (int)blockIdx.y;
    int cls = (int)blockIdx.z;
    int acc = 0;
    if (sample < n && cls < n_slots && clause < n_clauses) {
        int id = cls * n_clauses + clause;
        const unsigned char *base = patch_fired
            + ((unsigned long long)id * (unsigned)n + (unsigned)sample) * (unsigned)n_patches;
        int chosen = 0;
        int count = 0;
        unsigned rng = tc_hash(
            seed ^ (unsigned)(id + 1) * 0x9e3779b9u ^ (unsigned)(sample + 1) * 0xc2b2ae35u);
        if (rng == 0u) {
            rng = 1u;
        }
        for (int patch = 0; patch < n_patches; ++patch) {
            if (base[patch]) {
                ++count;
                rng = tc_xorshift(rng);
                if ((rng % (unsigned)count) == 0u) {
                    chosen = patch;
                }
            }
        }
        int vote = count > 0;
        fired[(unsigned long long)id * (unsigned)n + (unsigned)sample] = (unsigned char)vote;
        patch_index[(unsigned long long)id * (unsigned)n + (unsigned)sample] = chosen;
        if (vote) {
            acc = weights[id];
        }
    }
    int total = tc_block_sum(acc);
    if (threadIdx.x == 0 && sample < n && cls < n_slots) {
        atomicAdd(sums + (unsigned long long)sample * (unsigned)n_slots + (unsigned)cls, total);
    }
}

/*
Pack image patches into the same bit layout as tmu_encode. One thread owns
one patch of one example and writes only that row, so the rows stay in
example-major order. Neighboring patches are packed separately; the matcher
reads those packed rows with tensor cores instead of gathering windows on
the host.
*/
__global__ void tc_encode_patches(
    const unsigned int *X,
    unsigned int *encoded,
    int n_examples,
    int dim_x,
    int dim_y,
    int dim_z,
    int patch_w,
    int patch_h,
    int append_negated,
    int class_features)
{
    int n_px = dim_x - patch_w + 1;
    int n_py = dim_y - patch_h + 1;
    int n_patches = n_px * n_py;
    if (n_px < 1 || n_py < 1) {
        return;
    }
    int features = class_features
        + patch_w * patch_h * dim_z
        + (dim_x - patch_w)
        + (dim_y - patch_h);
    int chunks = append_negated
        ? ((2 * features - 1) / 32 + 1)
        : ((features - 1) / 32 + 1);
    unsigned long long job = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)n_examples * (unsigned)n_patches;
    if (job >= total) {
        return;
    }
    int example = (int)(job / (unsigned)n_patches);
    int patch = (int)(job - (unsigned long long)example * (unsigned)n_patches);
    int y = patch / n_px;
    int x = patch - y * n_px;
    const unsigned int *Xi = X
        + (unsigned long long)example * (unsigned)(dim_x * dim_y * dim_z);
    unsigned int *out = encoded
        + ((unsigned long long)example * (unsigned)n_patches + (unsigned)patch) * (unsigned)chunks;
    for (int c = 0; c < chunks; ++c) {
        out[c] = 0u;
    }
    for (int class_feature = 0; class_feature < class_features; ++class_feature) {
        int lit = class_feature + features;
        out[lit / 32] |= 1u << (lit % 32);
    }
    int y_span = dim_y - patch_h;
    for (int y_threshold = 0; y_threshold < y_span; ++y_threshold) {
        int patch_pos = class_features + y_threshold;
        if (y > y_threshold) {
            out[patch_pos / 32] |= 1u << (patch_pos % 32);
        } else if (append_negated) {
            int lit = patch_pos + features;
            out[lit / 32] |= 1u << (lit % 32);
        }
    }
    int x_span = dim_x - patch_w;
    for (int x_threshold = 0; x_threshold < x_span; ++x_threshold) {
        int patch_pos = class_features + y_span + x_threshold;
        if (x > x_threshold) {
            out[patch_pos / 32] |= 1u << (patch_pos % 32);
        } else if (append_negated) {
            int lit = patch_pos + features;
            out[lit / 32] |= 1u << (lit % 32);
        }
    }
    int content0 = class_features + y_span + x_span;
    for (int p_y = 0; p_y < patch_h; ++p_y) {
        for (int p_x = 0; p_x < patch_w; ++p_x) {
            for (int z = 0; z < dim_z; ++z) {
                int image_pos = (y + p_y) * dim_x * dim_z + (x + p_x) * dim_z + z;
                int patch_pos = content0 + p_y * patch_w * dim_z + p_x * dim_z + z;
                if (Xi[image_pos] == 1u) {
                    out[patch_pos / 32] |= 1u << (patch_pos % 32);
                } else if (append_negated) {
                    int lit = patch_pos + features;
                    out[lit / 32] |= 1u << (lit % 32);
                }
            }
        }
    }
}

/*
The live-word lists sit after the eight clause bitsets. flag[chunks] is
byte packed, then the int list is 4-byte aligned and holds chunks entries
plus the count. conv_live_index_bytes in the host uses this same layout.
*/
__device__ __forceinline__ void tc_conv_live_index(
    int bit_words,
    int chunks,
    unsigned char *&flag,
    int *&live,
    int *&nlive)
{
    unsigned char *base = reinterpret_cast<unsigned char *>(tc_dyn);
    int bitset_bytes = 8 * bit_words * (int)sizeof(unsigned);
    flag = base + bitset_bytes;
    int live_at = (bitset_bytes + chunks + 3) & ~3;
    live = reinterpret_cast<int *>(base + live_at);
    nlive = live + chunks;
}

/*
One warp owns 16 patches that start on a multiple of 16, so they share one
bitset word. Each lane ORs its own rows, then the lanes of one clause combine
with a warp shuffle and publish a single word.
*/
__device__ __forceinline__ void tc_conv_store_tile(
    unsigned *sh_bits,
    int bit_words,
    int patch_base,
    int g,
    int t,
    int score0,
    int score1,
    int score2,
    int score3)
{
    int row0 = patch_base + g;
    unsigned bits_lo = 0u;
    unsigned bits_hi = 0u;
    if (score0) bits_lo |= 1u << (row0 & 31);
    if (score1) bits_hi |= 1u << (row0 & 31);
    int row1 = row0 + 8;
    if (score2) bits_lo |= 1u << (row1 & 31);
    if (score3) bits_hi |= 1u << (row1 & 31);
    #pragma unroll
    for (int offset = 4; offset <= 16; offset <<= 1) {
        bits_lo |= __shfl_xor_sync(0xffffffffu, bits_lo, offset);
        bits_hi |= __shfl_xor_sync(0xffffffffu, bits_hi, offset);
    }
    if (g == 0) {
        int word = patch_base >> 5;
        if (bits_lo) atomicOr(sh_bits + (t * 2) * bit_words + word, bits_lo);
        if (bits_hi) atomicOr(sh_bits + (t * 2 + 1) * bit_words + word, bits_hi);
    }
}

__device__ __forceinline__ unsigned tc_conv_rng(
    unsigned seed, int cls, int n_clauses, int clause, int image)
{
    unsigned rng = tc_hash(
        seed ^ (unsigned)(cls * n_clauses + clause + 1) * 0x9e3779b9u
        ^ (unsigned)(image + 1) * 0xc2b2ae35u);
    return rng == 0u ? 1u : rng;
}

/*
Reservoir over the matching patches, in patch order. all_patches visits every
patch, which is what an empty include mask does. Prediction only needs to know
that one patch matched, so it stops at the first set bit.
*/
__device__ __forceinline__ int tc_conv_panel_vote(
    int warp,
    int g,
    int t,
    int panel,
    int cls,
    int image,
    int n,
    int n_clauses,
    int n_patches,
    int bit_words,
    int write_outputs,
    int all_patches,
    const unsigned *sh_bits,
    unsigned char *fired,
    int *patch_index,
    const int *cweights,
    unsigned seed)
{
    int acc = 0;
    int lane = (g << 2) | t;
    if (warp != 0 || lane >= 8) {
        return acc;
    }
    {
        int local = lane;
        int clause = panel * 8 + local;
        if (clause >= n_clauses) {
            return acc;
        }
        int weight = __ldg(cweights + clause);
        if (all_patches && !write_outputs) {
            acc += weight;
        } else if (!all_patches && !write_outputs) {
            const unsigned *clause_bits = sh_bits + local * bit_words;
            int hit = 0;
            for (int w = 0; w < bit_words && !hit; ++w) {
                unsigned bits = clause_bits[w];
                int valid = n_patches - (w << 5);
                if (valid <= 0) {
                    break;
                }
                if (valid < 32) {
                    bits &= (1u << valid) - 1u;
                }
                hit = bits != 0u;
            }
            if (hit) {
                acc += weight;
            }
        } else {
            int count = 0;
            int chosen = 0;
            unsigned rng = tc_conv_rng(seed, cls, n_clauses, clause, image);
            if (all_patches) {
                for (int patch = 0; patch < n_patches; ++patch) {
                    ++count;
                    rng = tc_xorshift(rng);
                    if ((rng % (unsigned)count) == 0u) {
                        chosen = patch;
                    }
                }
            } else {
                const unsigned *clause_bits = sh_bits + local * bit_words;
                for (int w = 0; w < bit_words; ++w) {
                    unsigned bits = clause_bits[w];
                    int base = w << 5;
                    int valid = n_patches - base;
                    if (valid <= 0) {
                        break;
                    }
                    if (valid < 32) {
                        bits &= (1u << valid) - 1u;
                    }
                    while (bits) {
                        int bit = __ffs(bits) - 1;
                        bits &= bits - 1;
                        ++count;
                        rng = tc_xorshift(rng);
                        if ((rng % (unsigned)count) == 0u) {
                            chosen = base + bit;
                        }
                    }
                }
            }
            if (write_outputs) {
                unsigned long long id =
                    ((unsigned long long)cls * (unsigned)n_clauses + (unsigned)clause) * (unsigned)n
                    + (unsigned)image;
                fired[id] = (unsigned char)(count > 0);
                patch_index[id] = chosen;
            }
            if (count > 0) {
                acc += weight;
            }
        }
    }
    return acc;
}

__device__ __forceinline__ unsigned tc_conv_load(
    const unsigned int *x_global,
    const unsigned int *x_shared,
    int use_shared,
    int row,
    int word,
    int chunks,
    int n_patches)
{
    if (row >= n_patches || word < 0) {
        return 0u;
    }
    unsigned long long offset =
        (unsigned long long)row * (unsigned)chunks + (unsigned)word;
    if (use_shared) {
        return x_shared[offset];
    }
    return __ldg(x_global + offset);
}

/*
Convolution match. One block owns one panel of eight clauses and ORs every
patch of one image. The patch grid is wider than one tensor-core fragment,
so the warps stride across it. The image stays in device memory: copying it
into every block repeats the load and includes words a long row will skip.
Masks for a short literal row are loaded once per panel. A firing clause
keeps one uniformly chosen matching patch for feedback.
grid: (panels, classes, images). X is [image, patch, chunk].
*/
__global__ void tc_conv_bmma(
    const unsigned int *X,
    const unsigned int *masks,
    const int *counts,
    const int *weights,
    unsigned char *fired,
    int *patch_index,
    int *sums,
    int n,
    int n_clauses,
    int n_slots,
    int chunks,
    int n_patches,
    int reject_empty,
    int write_outputs,
    int bit_words,
    int panels_per,
    int prefix_words,
    int x_words,
    unsigned seed)
{
    // Clauses are strided by bit_words. Each panel clears the whole bitset,
    // so one panel cannot inherit the previous panel's matches.
    unsigned *sh_bits = tc_dyn;
    int panel_count = (n_clauses + 7) >> 3;
    int panel_begin = (int)blockIdx.x * panels_per;
    if (bit_words < 1 || panel_begin >= panel_count || panels_per < 1) {
        return;
    }
    int panel_end = panel_begin + panels_per;
    if (panel_end > panel_count) {
        panel_end = panel_count;
    }
    int cls = (int)blockIdx.y;
    int image = (int)blockIdx.z;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    const int *ccounts = counts + cls * n_clauses;
    const int *cweights = weights + cls * n_clauses;
    const unsigned int *x_image = X
        + ((unsigned long long)image * (unsigned)n_patches) * (unsigned)chunks;
    unsigned *sh_x = tc_dyn + prefix_words;
    int use_shared = x_words > 0;
    if (use_shared) {
        for (int i = threadIdx.x; i < x_words; i += blockDim.x) {
            sh_x[i] = __ldg(x_image + i);
        }
        __syncthreads();
    }
    unsigned b0s[4];
    unsigned b1s[4];
    int acc = 0;
    for (int panel = panel_begin; panel < panel_end; ++panel) {
        int all_full = reject_empty ? 0 : 1;
        if (all_full) {
            for (int local = 0; local < 8; ++local) {
                int clause = panel * 8 + local;
                if (clause < n_clauses && __ldg(ccounts + clause) != 0) {
                    all_full = 0;
                }
            }
        }
        if (all_full) {
            acc += tc_conv_panel_vote(
                warp, g, t, panel, cls, image, n, n_clauses, n_patches, bit_words,
                write_outputs, 1, sh_bits, fired, patch_index, cweights, seed);
        } else {
            int c0 = panel * 8 + t * 2;
            int c1 = c0 + 1;
            int raw0 = (c0 < n_clauses) ? __ldg(ccounts + c0) : -1;
            int raw1 = (c1 < n_clauses) ? __ldg(ccounts + c1) : -1;
            int need0 = raw0;
            int need1 = raw1;
            if (reject_empty) {
                if (need0 == 0) need0 = -1;
                if (need1 == 0) need1 = -1;
            }
            for (int i = threadIdx.x; i < 8 * bit_words; i += blockDim.x) {
                sh_bits[i] = 0u;
            }
            __syncthreads();
            int bclause = panel * 8 + g;
            int ntiles = (chunks + 7) >> 3;
            int hoist = ntiles > 0 && ntiles <= 4;
            if (hoist && bclause < n_clauses) {
                const unsigned int *bp = masks
                    + ((unsigned long long)cls * (unsigned)n_clauses + (unsigned)bclause)
                      * (unsigned)chunks;
                for (int k = 0; k < ntiles; ++k) {
                    int word0 = k * 8 + t;
                    int word1 = word0 + 4;
                    b0s[k] = (word0 < chunks) ? __ldg(bp + word0) : 0u;
                    b1s[k] = (word1 < chunks) ? __ldg(bp + word1) : 0u;
                }
            }
            for (int patch_base = warp * 16; patch_base < n_patches; patch_base += nwarps * 16) {
                int row0 = patch_base + g;
                int row1 = row0 + 8;
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                for (int tile = 0, k = 0; tile < chunks; tile += 8, ++k) {
                    int word0 = tile + t;
                    int word1 = tile + 4 + t;
                    unsigned a0 = tc_conv_load(x_image, sh_x, use_shared, row0, word0 < chunks ? word0 : -1, chunks, n_patches);
                    unsigned a1 = tc_conv_load(x_image, sh_x, use_shared, row1, word0 < chunks ? word0 : -1, chunks, n_patches);
                    unsigned a2 = tc_conv_load(x_image, sh_x, use_shared, row0, word1 < chunks ? word1 : -1, chunks, n_patches);
                    unsigned a3 = tc_conv_load(x_image, sh_x, use_shared, row1, word1 < chunks ? word1 : -1, chunks, n_patches);
                    unsigned b0 = 0u;
                    unsigned b1 = 0u;
                    if (hoist && bclause < n_clauses) {
                        b0 = b0s[k];
                        b1 = b1s[k];
                    } else if (bclause < n_clauses) {
                        const unsigned int *bp = masks
                            + ((unsigned long long)cls * (unsigned)n_clauses + (unsigned)bclause)
                              * (unsigned)chunks;
                        if (word0 < chunks) b0 = __ldg(bp + word0);
                        if (word1 < chunks) b1 = __ldg(bp + word1);
                    }
                    tc_bmma_op(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                }
                int score0 = (c0 < n_clauses && row0 < n_patches && d0 == need0);
                int score1 = (c1 < n_clauses && row0 < n_patches && d1 == need1);
                int score2 = (c0 < n_clauses && row1 < n_patches && d2 == need0);
                int score3 = (c1 < n_clauses && row1 < n_patches && d3 == need1);
                tc_conv_store_tile(sh_bits, bit_words, patch_base, g, t, score0, score1, score2, score3);
            }
            __syncthreads();
            acc += tc_conv_panel_vote(
                warp, g, t, panel, cls, image, n, n_clauses, n_patches, bit_words,
                write_outputs, 0, sh_bits, fired, patch_index, cweights, seed);
        }
        __syncthreads();
    }
    int total = tc_block_sum(acc);
    if (threadIdx.x == 0 && image < n && cls < n_slots) {
        atomicAdd(sums + (unsigned long long)image * (unsigned)n_slots + (unsigned)cls, total);
    }
}

/*
Same convolution as tc_conv_bmma for a long literal row. The eight clauses
build the words any of them includes. The host places that list after the
patch bitset and launches this kernel when both fit in shared memory. The
panel uses the list when it is at most half the row, so a channel stack does
not drag every empty plane through the tensor cores. A panel that still
touches most words keeps the dense tiles.
*/
__global__ void tc_conv_bmma_live(
    const unsigned int *X,
    const unsigned int *masks,
    const int *counts,
    const int *weights,
    unsigned char *fired,
    int *patch_index,
    int *sums,
    int n,
    int n_clauses,
    int n_slots,
    int chunks,
    int n_patches,
    int reject_empty,
    int write_outputs,
    int bit_words,
    int panels_per,
    int prefix_words,
    int x_words,
    unsigned seed)
{
    unsigned *sh_bits = tc_dyn;
    unsigned char *sh_flag = nullptr;
    int *sh_live = nullptr;
    int *nlive_out = nullptr;
    int panel_count = (n_clauses + 7) >> 3;
    int panel_begin = (int)blockIdx.x * panels_per;
    if (bit_words < 1 || panel_begin >= panel_count || panels_per < 1) {
        return;
    }
    int panel_end = panel_begin + panels_per;
    if (panel_end > panel_count) {
        panel_end = panel_count;
    }
    tc_conv_live_index(bit_words, chunks, sh_flag, sh_live, nlive_out);
    int cls = (int)blockIdx.y;
    int image = (int)blockIdx.z;
    int warp = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    const int *ccounts = counts + cls * n_clauses;
    const int *cweights = weights + cls * n_clauses;
    const unsigned int *x_image = X
        + ((unsigned long long)image * (unsigned)n_patches) * (unsigned)chunks;
    unsigned *sh_x = tc_dyn + prefix_words;
    int use_shared = x_words > 0;
    if (use_shared) {
        for (int i = threadIdx.x; i < x_words; i += blockDim.x) {
            sh_x[i] = __ldg(x_image + i);
        }
        __syncthreads();
    }
    unsigned b0s[4];
    unsigned b1s[4];
    int acc = 0;
    int indexed = chunks > 0;
    for (int panel = panel_begin; panel < panel_end; ++panel) {
        int all_full = reject_empty ? 0 : 1;
        if (all_full) {
            for (int local = 0; local < 8; ++local) {
                int clause = panel * 8 + local;
                if (clause < n_clauses && __ldg(ccounts + clause) != 0) {
                    all_full = 0;
                }
            }
        }
        if (all_full) {
            acc += tc_conv_panel_vote(
                warp, g, t, panel, cls, image, n, n_clauses, n_patches, bit_words,
                write_outputs, 1, sh_bits, fired, patch_index, cweights, seed);
        } else {
            int c0 = panel * 8 + t * 2;
            int c1 = c0 + 1;
            int raw0 = (c0 < n_clauses) ? __ldg(ccounts + c0) : -1;
            int raw1 = (c1 < n_clauses) ? __ldg(ccounts + c1) : -1;
            int need0 = raw0;
            int need1 = raw1;
            if (reject_empty) {
                if (need0 == 0) need0 = -1;
                if (need1 == 0) need1 = -1;
            }
            for (int i = threadIdx.x; i < 8 * bit_words; i += blockDim.x) {
                sh_bits[i] = 0u;
            }
            if (indexed) {
                for (int word = threadIdx.x; word < chunks; word += blockDim.x) {
                    sh_flag[word] = 0;
                }
            }
            __syncthreads();
            if (indexed) {
                for (int word = threadIdx.x; word < chunks; word += blockDim.x) {
                    unsigned any = 0u;
                    for (int local = 0; local < 8; ++local) {
                        int clause = panel * 8 + local;
                        if (clause < n_clauses) {
                            any |= __ldg(masks
                                + ((unsigned long long)cls * (unsigned)n_clauses + (unsigned)clause)
                                  * (unsigned)chunks
                                + (unsigned)word);
                        }
                    }
                    sh_flag[word] = (unsigned char)(any != 0u);
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    int found = 0;
                    for (int word = 0; word < chunks; ++word) {
                        if (sh_flag[word]) {
                            sh_live[found++] = word;
                        }
                    }
                    *nlive_out = found;
                }
                __syncthreads();
            } else if (threadIdx.x == 0) {
                *nlive_out = chunks;
            }
            if (!indexed) {
                __syncthreads();
            }
            int nlive = *nlive_out;
            int gather = indexed && nlive * 2 <= chunks;
            int limit = gather ? nlive : chunks;
            int bclause = panel * 8 + g;
            int ntiles = (limit + 7) >> 3;
            int hoist = ntiles > 0 && ntiles <= 4;
            if (hoist && bclause < n_clauses) {
                const unsigned int *bp = masks
                    + ((unsigned long long)cls * (unsigned)n_clauses + (unsigned)bclause)
                      * (unsigned)chunks;
                for (int k = 0; k < ntiles; ++k) {
                    int tile = k * 8;
                    int word0;
                    int word1;
                    if (gather) {
                        int slot0 = tile + t;
                        int slot1 = tile + 4 + t;
                        word0 = (slot0 < nlive) ? sh_live[slot0] : -1;
                        word1 = (slot1 < nlive) ? sh_live[slot1] : -1;
                    } else {
                        word0 = tile + t;
                        word1 = tile + 4 + t;
                        if (word0 >= chunks) word0 = -1;
                        if (word1 >= chunks) word1 = -1;
                    }
                    b0s[k] = (word0 >= 0) ? __ldg(bp + word0) : 0u;
                    b1s[k] = (word1 >= 0) ? __ldg(bp + word1) : 0u;
                }
            }
            for (int patch_base = warp * 16; patch_base < n_patches; patch_base += nwarps * 16) {
                int row0 = patch_base + g;
                int row1 = row0 + 8;
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                for (int tile = 0, k = 0; tile < limit; tile += 8, ++k) {
                    int word0;
                    int word1;
                    if (gather) {
                        int slot0 = tile + t;
                        int slot1 = tile + 4 + t;
                        word0 = (slot0 < nlive) ? sh_live[slot0] : -1;
                        word1 = (slot1 < nlive) ? sh_live[slot1] : -1;
                    } else {
                        word0 = tile + t;
                        word1 = tile + 4 + t;
                        if (word0 >= chunks) word0 = -1;
                        if (word1 >= chunks) word1 = -1;
                    }
                    unsigned a0 = tc_conv_load(x_image, sh_x, use_shared, row0, word0, chunks, n_patches);
                    unsigned a1 = tc_conv_load(x_image, sh_x, use_shared, row1, word0, chunks, n_patches);
                    unsigned a2 = tc_conv_load(x_image, sh_x, use_shared, row0, word1, chunks, n_patches);
                    unsigned a3 = tc_conv_load(x_image, sh_x, use_shared, row1, word1, chunks, n_patches);
                    unsigned b0 = 0u;
                    unsigned b1 = 0u;
                    if (hoist && bclause < n_clauses) {
                        b0 = b0s[k];
                        b1 = b1s[k];
                    } else if (bclause < n_clauses) {
                        const unsigned int *bp = masks
                            + ((unsigned long long)cls * (unsigned)n_clauses + (unsigned)bclause)
                              * (unsigned)chunks;
                        if (word0 >= 0) b0 = __ldg(bp + word0);
                        if (word1 >= 0) b1 = __ldg(bp + word1);
                    }
                    tc_bmma_op(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                }
                int score0 = (c0 < n_clauses && row0 < n_patches && d0 == need0);
                int score1 = (c1 < n_clauses && row0 < n_patches && d1 == need1);
                int score2 = (c0 < n_clauses && row1 < n_patches && d2 == need0);
                int score3 = (c1 < n_clauses && row1 < n_patches && d3 == need1);
                tc_conv_store_tile(sh_bits, bit_words, patch_base, g, t, score0, score1, score2, score3);
            }
            __syncthreads();
            acc += tc_conv_panel_vote(
                warp, g, t, panel, cls, image, n, n_clauses, n_patches, bit_words,
                write_outputs, 0, sh_bits, fired, patch_index, cweights, seed);
        }
        __syncthreads();
    }
    int total = tc_block_sum(acc);
    if (threadIdx.x == 0 && image < n && cls < n_slots) {
        atomicAdd(sums + (unsigned long long)image * (unsigned)n_slots + (unsigned)cls, total);
    }
}

}
