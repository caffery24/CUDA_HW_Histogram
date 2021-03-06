#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t *result, uint32_t *input[], int height, int width);

/* Include below the function headers of any other functions that you implement */

void copy_final(uint32_t **data, uint32_t *histo_bin, uint32_t *histo);

uint32_t ** setup_data(uint32_t ** input, int width, int height);

uint32_t * setup_histo();

#define NUM_HISTO 6 

#endif
