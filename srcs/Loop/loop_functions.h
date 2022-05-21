/**
*	---------------------------------------------------------------------
*	Author : Wayne
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#pragma once

#include "../core/basic.h"


namespace de
{
	class Stream
	{
	public:

		Stream();
		~Stream();
	};
}


de::Stream::Stream() noexcept
{
}

de::Stream::~Stream() noexcept
{
}




namespace decx
{
	class _Stream
	{
	public:
		cudaStream_t *stream;

		_Stream();
		~_Stream();

	};
}



decx::_Stream::_Stream() noexcept
{
	checkCudaErrors(cudaStreamCreate(this->stream));
}

decx::_Stream::~_Stream() noexcept
{
	checkCudaErrors(cudaStreamDestroy(*(this->stream)));
	printf("yes");
	//__crt_va_arg()
	std::thread a();
}