#ifndef CISS_H
#define CISS_H

//include
#include "io.h"
#include "sptensor.h"
#include "csf.h"

/*
This is used for loading tensors in MBLC
*/


/**
 * @brief the struct for ciss format
 * @param entries: the entries for index and tile indication + value
 * @param directories: guidance for another index
*/
typedef struct 
{
    idx_t nmodes;
    idx_t nnz;
    idx_t size; //the total size for entries
    idx_t dlength; //the length for directory
    idx_t * dims; //the actual dimension
    idx_t * directory; //list for the first dimension(mode-1)
    idx_t * dcounter; //for SGD
    double * entries; //actual elements(including indices and values)
}ciss_t;

//public function
//now for single gpu
ciss_t* ciss_alloc(
    sptensor_t * newtensor,
    idx_t mode
);

// A sparse tensor in BLCO format
typedef struct
{
    // The number of modes the tensor has
    idx_t N;

    // Length `N` array where each element is length of the corresponding mode
    idx_t* modes;

    // Length `N` array of masks per mode
    idx_t* mode_masks;

    // Length `N` array of shift offsets for modes
    int* mode_pos;

    // Same as `modes` but modes are number of bits required for each mode
    //IType* modes_bitcount = nullptr;

    // Total number of non-zeros across all blocks
    idx_t total_nnz;

    // Number of nonzero elements in largest block
    idx_t max_nnz;

    // The number of blocks in this tensor
    idx_t block_count;

    // The maximum number of nonzero elements in each block
    idx_t max_block_size;

    // Length `block_count` array, pointers to the blocks themselves
    // blco_block** blocks = nullptr;

    // Length `block_count` array, pointers to the blocks themselves, on the GPU
    // blco_block** blocks_dev_staging = nullptr;

    // Same as blocks_dev, but on the GPU
    // blco_block** blocks_dev_ptr = nullptr;

    // Length ceil(`nnz` / TILE_SIZE) array, block information for each tile
    idx_t* tile_info;

    // Length `total_nnz` array, the linearized coordinates of each value
    idx_t* coords;

    // Length `total_nnz` array, the values corresponding to each coordinate
    double* values;

    // Length `block_count` array, the GPU streams associated with each block
    // cudaStream_t* streams = nullptr;

    // Length `block_count` array, the GPU events/signals associated with each block
    //cudaEvent_t* events = nullptr;

    idx_t* warp_info;
    idx_t* warp_info_gpu;
    idx_t warp_info_length;
}blcotensor;

blcotensor* blco_alloc(
    sptensor_t * newtensor,
    idx_t mode
);

ciss_t* ciss_copy(
    ciss_t * oldtensor
);

void ciss_display(ciss_t* newtensor);

void ciss_free(ciss_t* newtensor);


#endif