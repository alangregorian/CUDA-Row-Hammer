//
// Created by aj on 6/26/24.
//

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>

#include <vector>

#include "../reverse_engineering.hpp"
//
//#include <fractional_gpu.hpp>
//#include <fractional_gpu_cuda.cuh>

/*
#if !defined(FGPU_TEST_MEM_COLORING_ENABLED)
#error "FGPU_TEST_MEM_COLORING_ENABLED not defined. Needed for reverse engineering"
#endif
 */

/* Device memory */
static uint64_t *d_sum;
static uint64_t *d_refresh_v;
static size_t max_custom_pchase_entires;
static uint64_t **d_custom_pchase;
static uint64_t **h_custom_pchase;
static double *d_ticks;
static uint64_t **d_last_addr;
static double *h_ticks;
static uint64_t *h_a;
static uint64_t *d_count;

/*
 * Read enough data to implicitly flush L2 cache.
 * Uses p-chase to make sure compiler/hardware doesn't optimize away the code.
 */
__device__
uint64_t refresh_l2(volatile uint64_t *refresh_vaddr)
{
    uint64_t curindex = 0;
    uint64_t sum = 0;

    while (curindex != (uint64_t)-1) {
        curindex = refresh_vaddr[curindex];
        sum += curindex;
    }
    return sum;
}

/*
 * Reads data from a_v and b_v arrays together and measure time. This can be
 * used to see if they both lie on same dram bank.
 * XXX: This function for some reason doesn't work correctly for all pairs
 * of a_v, b_v. It is giving false negatives.
 */
__global__
void read_dram_pair(volatile uint64_t *a_v, volatile uint64_t *b_v,
                    volatile uint64_t *refresh_v, volatile double *ticks, volatile uint64_t *psum,
                    double threshold)
{
    uint64_t curindex;
    uint64_t sum;
    uint64_t count;
    uint64_t mid;
    uint64_t previndex;
    const uint64_t sharednum = 1;
    __shared__ uint64_t s[sharednum];
    __shared__ uint64_t t[sharednum];
    uint64_t tsum;
    int i;
    double tick;

    for (i = 0; i < GPU_MAX_OUTER_LOOP + 1; i++) {

        sum = 0;
        curindex = 0;

        /* Evict all data from L2 cache */
        sum += refresh_l2(refresh_v);

        /*
         * Measure time to read two different addresses together. If lie on
         * same bank, different rows, we expect to see a jump in time
         */
        while (curindex != (uint64_t)-1) {
            previndex = curindex;
            mid = clock64();
            sum += b_v[curindex];
            curindex = a_v[curindex];
            s[previndex] = curindex;
            t[previndex] = clock64() - mid;
        }

        /* Some extra code to make sure hardware/compiler doesn't optimize code */
        curindex = 0;
        tsum = 0;
        count = 0;
        while (curindex != (uint64_t)-1) {
            count++;
            tsum += t[curindex];
            curindex = s[curindex];
        }

        /* First run is warmup - Effects like TLB miss might impact*/
        if (i == 0)
            continue;

        tick = ((double)tsum) / ((double)count);

        if (tick > threshold) {
            /* We don't expect threshold to be crossed on GPU (No timer interrupts) */
            printf("ERROR: Threshold:%f, Ticks:%f, i:%d, count: %ld\n", threshold, tick, i, count);
            i--;
            continue;
        }

        ticks[i - 1] = tick;
        psum[i - 1] = sum;
    }
}


/* Find the average time to read from arrays a and b together (from DRAM) */
double device_find_dram_read_time(void *_a, void *_b, double threshold)
{
    uint64_t *a = (uint64_t *)_a;
    uint64_t *b = (uint64_t *)_b;
    int i;
    double min_ticks, max_ticks, sum_ticks;

    cudaMemcpy(a, h_a, GPU_L2_CACHE_LINE_SIZE, cudaMemcpyHostToDevice);
    read_dram_pair<<<1,1>>>(a, b, d_refresh_v, d_ticks, d_sum, threshold);
    cudaDeviceSynchronize();
    cudaMemcpy(h_ticks, d_ticks, GPU_MAX_OUTER_LOOP * sizeof(double), cudaMemcpyDeviceToHost);

    for (i = 0, min_ticks = LONG_MAX, sum_ticks = 0, max_ticks = 0;
         i < GPU_MAX_OUTER_LOOP; i++) {
        double tick = h_ticks[i];

        assert(tick > 0);

        min_ticks = tick < min_ticks ? tick : min_ticks;
        max_ticks = tick > max_ticks ? tick : max_ticks;
        sum_ticks += tick;
    }

    dprintf("Min Ticks: %0.3f,\tAvg Ticks: %0.3f,\tMax Ticks: %0.3f\n",
            min_ticks, (sum_ticks * 1.0f) / GPU_MAX_OUTER_LOOP, max_ticks);
    /* Min ticks are more reliable source. Avg gets influences my outliers */
    return min_ticks;
}
