/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#include "MemBlock.h"
#include "internal_types.h"


class decx::MemChunk_Hv
{
public:
    uchar* header_ptr;
    size_t chunk_size;
    std::vector<decx::MemBlock*> mem_block_list;
    int list_length;

    /**
     * @brief Construct a new Mem Chunk object formally, it will not allocate the memory
     * physically, but it will turn the size into zero, and insert an ilde decx::MemBlock.(All params initialized)
     *
     * @param pool_dex The index of corresponding decx::MemChunkSet in memory pool.
     * @param chunk_set_dex The index of decx::MemChunk in decx::MemChunkSet.
     */
    MemChunk_Hv(int pool_dex, int chunk_set_dex);

    /**
     * @brief Construct a new Mem Chunk object, and it will allocate physically, then judge
     * if the block should be spilted.
     *
     * @param size The total size of the physical memory block, it is in 2's power.
     * @param req_size The required size of memory, given by users.
     * @param pool_dex The index of corresponding decx::MemChunkSet in memory pool.
     * @param chunk_set_dex The index of decx::MemChunk in decx::MemChunkSet.
     */
    MemChunk_Hv(size_t size, size_t req_size, int pool_dex, int chunk_set_dex);


    /**
     * @brief split the block indicated by the param 'dex', and refresh the decx::MemBlock::_loc.z
     * of all the following blocks.
     *
     * @param dex
     * @param split_size
     * @return decx::MemBlock* the pointer of the newly inserted decx::MemBlock
     */
    decx::MemBlock* split(int dex, size_t split_size);


    /**
     * @brief merge forward from the indicated decx::MemBlock, preserve the previous
     * block, and erase the indicated block.
     *
     * @param dex Where the merging operation starts
     * @return decx::MemBlock* the pointer of the former among the blocks being merged
     */
    decx::MemBlock* forward_merge2(int dex);


    /**
     * @brief merge backward from the indicated decx::MemBlock, preserve the indicated
     * block, and erase the next block.
     *
     * @param dex Where the merging operation starts
     * @return decx::MemBlock* the pointer of the former among the blocks being merged
     */
    decx::MemBlock* backward_merge2(int dex);



    decx::MemBlock* merge3(int dex);


    /**
     * @brief Search around the indicated block for any possibility to merge
     *
     * @param dex The index of the searching block
     */
    void check_to_merge(int dex);
};


decx::MemChunk_Hv::MemChunk_Hv(int pool_dex, int chunk_set_dex)
{
    this->chunk_size = 0;
    this->header_ptr = NULL;

    decx::MemLoc _loc;
    _loc.x = pool_dex;        _loc.y = chunk_set_dex;        _loc.z = 0;
    decx::MemBlock* new_node = new decx::MemBlock(0, true, &_loc, NULL, NULL, NULL);
    this->mem_block_list.emplace_back(new_node);

    this->list_length = this->mem_block_list.size();
}



decx::MemChunk_Hv::MemChunk_Hv(size_t size, size_t req_size, int pool_dex, int chunk_set_dex)
{
    this->chunk_size = size;

    uchar* ptr = (uchar*)decx::alloc::aligned_malloc_Hv(size, host_mem_alignment);
    this->header_ptr = ptr;
    int3 _loc;
    _loc.x = pool_dex;        _loc.y = chunk_set_dex;        _loc.z = 0;

    decx::MemBlock* new_node_0 = new decx::MemBlock(req_size, false, &_loc, ptr, NULL, NULL);
    this->mem_block_list.emplace_back(new_node_0);

    if (size != req_size) {
        _loc.z = 1;
        decx::MemBlock* new_node_1 = new decx::MemBlock(
            size - req_size, true, &_loc, ptr + req_size, NULL, NULL);

        new_node_0->_next = new_node_1;
        new_node_1->_prev = new_node_0;
        this->mem_block_list.emplace_back(new_node_1);
    }
    this->list_length = this->mem_block_list.size();
}



decx::MemBlock* decx::MemChunk_Hv::split(int dex, size_t req_size)
{
    decx::MemBlock* block_split = *(this->mem_block_list.begin() + dex);
    int3 _loc;
    _loc.x = block_split->_loc.x;
    _loc.y = block_split->_loc.y;
    _loc.z = block_split->_loc.z + 1;

    size_t splited_size = block_split->block_size - req_size;
    // create a new block
    decx::MemBlock* block_insert = new decx::MemBlock(
        splited_size, true, &_loc, block_split->_ptr + req_size, block_split, NULL);

    block_split->block_size = req_size;

    if (block_split->_next != NULL)
    {                                    // not the last block
        block_insert->_next = block_split->_next;
        for (int i = dex + 1; i < this->list_length; ++i) {
            this->mem_block_list[i]->_loc.z++;
        }
        this->mem_block_list.insert(this->mem_block_list.begin() + dex + 1, block_insert);
    }
    else {        // is the last block
        this->mem_block_list.emplace_back(block_insert);
    }

    block_split->_next = block_insert;
    return block_insert;
}



decx::MemBlock* decx::MemChunk_Hv::forward_merge2(int dex)
{
    decx::MemBlock* current_bl = *(this->mem_block_list.begin() + dex);
    decx::MemBlock* prev_bl = current_bl->_prev;
    // capacity merged
    prev_bl->block_size += current_bl->block_size;
    if (current_bl->_next != NULL) {        // not the last one
        decx::MemBlock* next_bl = current_bl->_next;
        next_bl->_prev = prev_bl;
        prev_bl->_next = next_bl;

        for (int i = dex + 1; i < this->list_length; ++i) {
            this->mem_block_list[i]->_loc.z--;
        }

        this->mem_block_list.erase(this->mem_block_list.begin() + dex);
        delete current_bl;
    }
    else {    // is the last block
        prev_bl->_next = NULL;
        this->mem_block_list.pop_back();
        delete current_bl;
    }

    return prev_bl;
}


decx::MemBlock* decx::MemChunk_Hv::backward_merge2(int dex)
{
    return decx::MemChunk_Hv::forward_merge2(dex + 1);
}


decx::MemBlock* decx::MemChunk_Hv::merge3(int dex)
{
    decx::MemBlock* this_bl = *(this->mem_block_list.begin() + dex);
    decx::MemBlock* prev_bl = this_bl->_prev;
    decx::MemBlock* next_bl = this_bl->_next;

    prev_bl->block_size += (this_bl->block_size + next_bl->block_size);

    if (next_bl->_next != NULL) {
        prev_bl->_next = next_bl->_next;
        next_bl->_next->_prev = prev_bl;

        for (int i = dex + 2; i < this->list_length; ++i) {
            this->mem_block_list[i]->_loc.z -= 2;
        }
        this->mem_block_list.erase(
            this->mem_block_list.begin() + dex, this->mem_block_list.begin() + dex + 2);

        delete this_bl;
        delete next_bl;
    }
    else {
        prev_bl->_next = NULL;
        this->mem_block_list.pop_back();
        this->mem_block_list.pop_back();

        delete this_bl;
        delete next_bl;
    }
    return prev_bl;
}


void decx::MemChunk_Hv::check_to_merge(int dex)
{
    decx::MemBlock* this_bl = *(this->mem_block_list.begin() + dex);
    // The last block, forward_merge2 only
    if (this_bl->_prev != NULL && this_bl->_next == NULL) {
        decx::MemBlock* prev_bl = this_bl->_prev;
        if (prev_bl->_idle) {
            this->forward_merge2(dex);
        }
    }
    // The first block, backward_merge only
    else if (this_bl->_prev == NULL && this_bl->_next != NULL) {
        decx::MemBlock* next_bl = this_bl->_next;
        if (next_bl->_idle) {
            this->backward_merge2(dex);
        }
    }
    else if (this_bl->_prev != NULL && this_bl->_next != NULL) {
        decx::MemBlock* prev_bl = this_bl->_prev;
        decx::MemBlock* next_bl = this_bl->_next;
        if (prev_bl->_idle && (!next_bl->_idle)) {
            this->forward_merge2(dex);
        }
        else if ((!prev_bl->_idle) && next_bl->_idle) {
            this->backward_merge2(dex);
        }
        else if (prev_bl->_idle && next_bl->_idle) {
            this->merge3(dex);
        }
    }
}


class decx::MemChunkSet_Hv
{
public:
    std::vector<decx::MemChunk_Hv> mem_chunk_list;
    size_t flag_size;
    int list_length;

    /**
     * @brief Construct a new Mem Chunk Set object, insert some initialized decx::MemBlock
     * to all the vector_list
     *
     * @param pool_dex The index of corresponding decx::MemChunkSet in memory pool.
     */
    MemChunkSet_Hv(int pool_dex);
};



decx::MemChunkSet_Hv::MemChunkSet_Hv(int pool_dex)
{
    this->mem_chunk_list.emplace_back(pool_dex, 0);
    this->flag_size = 1 << (dex_to_pow_bias + pool_dex);
    this->list_length = this->mem_chunk_list.size();
}
