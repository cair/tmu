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
    unsigned int seed)
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
                        unsigned x = X[(unsigned long long)s * (unsigned)chunks + chunk];
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
                    unsigned x = X[(unsigned long long)s * (unsigned)chunks + chunk];
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
                        unsigned x = X[(unsigned long long)s * (unsigned)chunks + chunk];
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
                    unsigned x = X[(unsigned long long)s * (unsigned)chunks + chunk];
                    tc_inc(plane, lit & ~x, bits);
                }
            }
            for (int b = 0; b < bits; ++b) base[b] = plane[b];
        }
    }
}

}
