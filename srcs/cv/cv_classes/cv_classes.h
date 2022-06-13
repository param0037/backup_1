/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/


#ifndef _CV_CLASS_H_
#define _CV_CLASS_H_

#include "../../core/basic.h"
#include "../../classes/classes_util.h"
#include "../../core/memory_management/MemBlock.h"
#include "../../core/allocators.h"



//#define IMG_ALIGNMENT_GRAY 16
//#define IMG_ALIGNMENT_RGBA 4

namespace de
{
    namespace vis
    {
        enum ImgConstructType
        {
            DE_UC1 = 1,
            DE_UC3 = 3,
            DE_UC4 = 4,
            DE_IMG_DEFAULT = 5,
            DE_IMG_4_ALIGN = 6
        };

        class _DECX_API_ Img
        {
        public:
            Img() {}

            virtual uint Width() { return 0; }

            virtual uint Height() { return 0; }

            virtual uchar* Ptr(const uint row, const uint col) {
                uchar* ptr = NULL;
                return ptr;
            }

            virtual void release() {
                return;
            }
        };
    }
}




namespace decx
{
    class _Img : public de::vis::Img
    {
    public:
        size_t ImgPlane, element_num, total_bytes;
        uint channel;

        int Mem_Store_Type, Image_Type;
        uint pitch;
        size_t _element_num, _total_bytes;

        decx::PtrInfo<uchar> Mat;
        uint width, height;

        _Img() {}


        _Img(const uint width, const uint heght, const int flag);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uchar* Ptr(const uint row, const uint col);


        virtual void release();
    };

}


/*
* I choose __m128 to load the uchar image array instead of __m256. If I use __m256 as the stride,
* register will be overloaded
*/
#define _IMG_ALIGN_ 32
#define _IMG_ALIGN4_ 8




decx::_Img::_Img(const uint width, const uint height, const int flag)
{
    this->height = height;
    this->width = width;
    this->ImgPlane = static_cast<size_t>(this->width) * static_cast<size_t>(this->height);

    switch (flag)
    {
    case de::vis::ImgConstructType::DE_UC1:        // UC1
        this->channel = 1;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;
        break;

    case de::vis::ImgConstructType::DE_UC3:        // UC3
        this->channel = 4;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN4_) * _IMG_ALIGN4_;
        break;

    case de::vis::ImgConstructType::DE_UC4:        // UC4
        this->channel = 4;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN4_) * _IMG_ALIGN4_;
        break;

    default:    // default situation : align up with 4 bytes(uchar4)
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;
        this->channel = 4;
        break;
    }

    this->element_num = static_cast<size_t>(this->channel) * this->ImgPlane;
    this->total_bytes = this->element_num * sizeof(uchar);
    this->_element_num = (size_t)this->pitch * (size_t)this->height * (size_t)this->channel;
    this->_total_bytes = this->_element_num * sizeof(uchar);

#ifndef GNU_CPUcodes
    if (decx::alloc::_host_virtual_page_malloc(&this->Mat, this->_total_bytes)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
#else
    if (decx::alloc::_host_virtual_page_malloc(&this->Mat, this->total_bytes)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
#endif
}



uchar* decx::_Img::Ptr(const uint row, const uint col)
{
    return (this->Mat.ptr + ((size_t)row * (size_t)(this->pitch) + col) * (size_t)(this->channel));
}





void decx::_Img::release()
{
#ifndef GNU_CPUcodes
    decx::alloc::_host_virtual_page_dealloc(&this->Mat);
#else
    decx::alloc::_host_virtual_page_dealloc(&this->Mat);
#endif
}



#endif