#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "mpi.h"
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// Uncomment to generate gif animation
// #define GENERATE_GIF

#define IDX(r, c, cols) ((r) * (cols) + (c))

inline double gauss(double x, double mu, double sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma);
}

double *generate_kernel(double *K, const unsigned int size)
{
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0.0;

    for (int y = -r; y < r; y++) {
        for (int x = -r; x < r; x++) {
            double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
            K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
            if (distance > 1)
                K[(y + r) * size + x + r] = 0.0;
            sum += K[(y + r) * size + x + r];
        }
    }

    for (unsigned int y = 0; y < size; y++)
        for (unsigned int x = 0; x < size; x++)
            K[y * size + x] /= sum;

    return K;
}

/*
 * Send bottom halo rows down, receive top halo rows from above.
 * Send top halo rows up,   receive bottom halo rows from below.
 * Wraps around (periodic boundary) via modular rank arithmetic.
 *
 * local_world layout:
 *   rows 0            .. halo-1          : top ghost rows    (filled by Sendrecv)
 *   rows halo         .. halo+local_rows-1: real data
 *   rows halo+local_rows .. halo+local_rows+halo-1: bottom ghost rows (filled by Sendrecv)
 */
void exchange_halos(
    double *local_world,
    unsigned int local_rows,
    unsigned int cols,
    unsigned int halo,
    int rank,
    int size
)
{
    unsigned int filled = 0;
    unsigned int hop = 1;

    while (filled < halo) {
        unsigned int chunk = local_rows < (halo - filled) ? local_rows : (halo - filled);

        int up   = (rank - hop + size * (hop + 1)) % size;
        int down = (rank + hop) % size;

        /* Fill top ghost zone from above, bottom ghost zone from below */
        MPI_Sendrecv(
            /* send our bottom real rows (the chunk closest to bottom) downward */
            &local_world[IDX(halo + local_rows - chunk, 0, cols)],
            chunk * cols, MPI_DOUBLE, down, hop,
            /* receive into top ghost zone, filling from center outward */
            &local_world[IDX(halo - filled - chunk, 0, cols)],
            chunk * cols, MPI_DOUBLE, up, hop,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        MPI_Sendrecv(
            /* send our top real rows (the chunk closest to top) upward */
            &local_world[IDX(halo, 0, cols)],
            chunk * cols, MPI_DOUBLE, up, hop + size,
            /* receive into bottom ghost zone, filling from center outward */
            &local_world[IDX(halo + local_rows + filled, 0, cols)],
            chunk * cols, MPI_DOUBLE, down, hop + size,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        filled += chunk;
        hop++;
    }
}

/*
 * Convolve a single output cell (local_i, j).
 * local_i is an index into the real rows (0..local_rows-1).
 * The halo offset is added internally.
 * Column wrapping is handled with modular arithmetic.
 */
double convolve_local_cell(
    const double *local_world,
    const double *w,
    unsigned int cols,
    unsigned int kernel_size,
    unsigned int halo,
    unsigned int local_i,
    unsigned int j
)
{
    double sum = 0.0;
    unsigned int center_i = local_i + halo;

    for (int ki = kernel_size - 1, kri = 0; ki >= 0; ki--, kri++) {
        for (int kj = kernel_size - 1, kcj = 0; kj >= 0; kj--, kcj++) {
            int local_row = (int)center_i - (int)halo + kri;
            int col = ((int)j - (int)halo + kcj + (int)cols) % (int)cols;
            sum += w[IDX(ki, kj, kernel_size)]
                 * local_world[IDX(local_row, col, cols)];
        }
    }
    return sum;
}

double *evolve_lenia(
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int steps,
    const double dt,
    const unsigned int kernel_size,
    const struct orbium_coo *orbiums,
    const unsigned int num_orbiums
)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rows % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: rows (%u) must be divisible by number of MPI processes (%d).\n", rows, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    unsigned int local_rows = rows / size;
    unsigned int halo       = kernel_size / 2;


    unsigned int local_size            = local_rows * cols;
    unsigned int local_with_halo_size  = (local_rows + 2 * halo) * cols;

    /* Kernel — generated on rank 0 and broadcast */
    double *w     = calloc(kernel_size * kernel_size, sizeof(double));
    double *world = NULL;   /* full grid, rank 0 only */

    double *local_world = calloc(local_with_halo_size, sizeof(double));
    double *local_tmp   = calloc(local_with_halo_size, sizeof(double));

    if (rank == 0) {
        world = calloc(rows * cols, sizeof(double));
        generate_kernel(w, kernel_size);
        for (unsigned int o = 0; o < num_orbiums; o++)
            world = place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
    }

    /* Share kernel and initial grid */
    MPI_Bcast(w, kernel_size * kernel_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(
        world, local_size, MPI_DOUBLE,
        &local_world[IDX(halo, 0, cols)], local_size, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

#ifdef GENERATE_GIF
    ge_GIF *gif = NULL;
    if (rank == 0) {
        gif = ge_new_gif(
            "lenia.gif", cols, rows,
            inferno_pallete, 8, -1, 0
        );
    }
#endif

    for (unsigned int step = 0; step < steps; step++) {

        /* Fill ghost rows from neighbours */
        exchange_halos(local_world, local_rows, cols, halo, rank, size);

        /* Convolve + grow */
        for (unsigned int local_i = 0; local_i < local_rows; local_i++) {
            for (unsigned int j = 0; j < cols; j++) {
                double u = convolve_local_cell(
                    local_world, w, cols, kernel_size, halo, local_i, j
                );
                unsigned int real_i = local_i + halo;
                double value = local_world[IDX(real_i, j, cols)]
                             + dt * growth_lenia(u);
                value = fmin(1.0, fmax(0.0, value));
                local_tmp[IDX(real_i, j, cols)] = value;
            }
        }

        /* Swap buffers (only real region changed; halos are refilled next iteration) */
        double *swap = local_world;
        local_world  = local_tmp;
        local_tmp    = swap;

        /* Collect full grid at root for GIF frame */
#ifdef GENERATE_GIF
        MPI_Gather(
            &local_world[IDX(halo, 0, cols)], local_size, MPI_DOUBLE,
            world, local_size, MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        if (rank == 0) {
            for (unsigned int i = 0; i < rows * cols; i++)
                gif->frame[i] = (uint8_t)(world[i] * 255);
            ge_add_frame(gif, 5);
        }
#endif
    }

    /* Final gather so rank 0 can return the complete world */
#ifndef GENERATE_GIF
    MPI_Gather(
        &local_world[IDX(halo, 0, cols)], local_size, MPI_DOUBLE,
        world, local_size, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );
#endif

#ifdef GENERATE_GIF
    if (rank == 0)
        ge_close_gif(gif);
#endif

    free(w);
    free(local_world);
    free(local_tmp);

    return rank == 0 ? world : NULL;
}