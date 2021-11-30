#include "NeuronStorages.h"

#include <algorithm>
#include <assert.h>
#include <xmmintrin.h>
#include <emmintrin.h>

SNeuronMatrixView::SNeuronMatrixView(float *data, size_t rows, size_t col, size_t rowStride)
:	m_Data(data)
,	m_Rows(rows)
,	m_Columns(col)
,	m_RowByteStride(rowStride)
{
}

SNeuronMatrixView::~SNeuronMatrixView()
{

}

float	*SNeuronMatrixView::GetRow(size_t idx) const
{
	return (float*)((char*)m_Data + idx * m_RowByteStride);
}

SConstNeuronMatrixView::SConstNeuronMatrixView(const float *data, size_t rows, size_t col, size_t rowStride)
:	 m_Data(data)
,	 m_Rows(rows)
,	 m_Columns(col)
,	 m_RowByteStride(rowStride)
{
}

SConstNeuronMatrixView::SConstNeuronMatrixView(const SNeuronMatrixView &oth)
:	 m_Data(oth.m_Data)
,	 m_Rows(oth.m_Rows)
,	 m_Columns(oth.m_Columns)
,	 m_RowByteStride(oth.m_RowByteStride)
{
}

SConstNeuronMatrixView::~SConstNeuronMatrixView()
{

}

const float	*SConstNeuronMatrixView::GetRow(size_t idx) const
{
	return (const float*)((const char*)m_Data + idx * m_RowByteStride);
}

CNeuronVector::CNeuronVector()
:	m_Data(nullptr)
,	m_Size(0)
{

}

CNeuronVector::~CNeuronVector()
{
	if (m_Data != nullptr)
		_aligned_free(m_Data);
}

bool	CNeuronVector::AllocateStorage(size_t elements)
{
	if (m_Data != nullptr)
		_aligned_free(m_Data);
	m_Size = elements;
	m_Data = (float*)_aligned_malloc(elements * sizeof(float), 0x10);
	return m_Data != nullptr;
}

CNeuronMatrix::CNeuronMatrix()
:	m_Mat(nullptr, 0, 0, 0)
{
}

CNeuronMatrix::~CNeuronMatrix()
{
	if (m_Mat.m_Data != nullptr)
		_aligned_free(m_Mat.m_Data);
}

bool	CNeuronMatrix::AllocMatrix(size_t rows, size_t col)
{
	const size_t	alignment = 0x10;
	size_t			colByteSize = col * sizeof(float);
	size_t			offsetToAlign = colByteSize & (alignment - 1);
	size_t			alignedColSize = offsetToAlign == 0 ? colByteSize : colByteSize + offsetToAlign;

	if (m_Mat.m_Data != nullptr)
		_aligned_free(m_Mat.m_Data);
	m_Mat.m_Data = (float*)_aligned_malloc(alignedColSize * rows, alignment);
	m_Mat.m_Rows = rows;
	m_Mat.m_Columns = col;
	m_Mat.m_RowByteStride = alignedColSize;
	return m_Mat.m_Data != nullptr;
}

template<bool _DstAligned, bool _SrcAligned, bool _AddAligned>
void	_ComputeNetInput(float *dst, const float *src, const SConstNeuronMatrixView& mul, const float *add)
{
#if		1
	// Reference non-SIMD code:
	for (size_t y = 0; y < mul.m_Rows; ++y)
	{
		float	sum = 0;
		for (size_t x = 0; x < mul.m_Columns; ++x)
		{
			sum += src[x] * mul.GetRow(y)[x];
		}
		dst[y] = sum + add[y];
	}
	return;
#endif
	float		*dstPtr = dst;
	const float	*addPtr = add;
	const float	*dstPtrStop = dst + mul.m_Rows;
	const float	*mulPtr = mul.m_Data;

	__m128		simdLastValuesMask[3];

	simdLastValuesMask[0] = _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, 0xFFFFFFFF));
	simdLastValuesMask[1] = _mm_castsi128_ps(_mm_set_epi32(0, 0, 0xFFFFFFFF, 0xFFFFFFFF));
	simdLastValuesMask[2] = _mm_castsi128_ps(_mm_set_epi32(0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF));

	__m128		simdMasksKeepOne[4];

	simdMasksKeepOne[0] = _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, 0xFFFFFFFF));
	simdMasksKeepOne[1] = _mm_castsi128_ps(_mm_set_epi32(0, 0, 0xFFFFFFFF, 0));
	simdMasksKeepOne[2] = _mm_castsi128_ps(_mm_set_epi32(0, 0xFFFFFFFF, 0, 0));
	simdMasksKeepOne[3] = _mm_castsi128_ps(_mm_set_epi32(0xFFFFFFFF, 0, 0, 0));

	dstPtrStop -= 4;
	while (dstPtr <= dstPtrStop)
	{
		__m128			accum_xyzw = _mm_set1_ps(0.0f);

		// This loop should be unrolled:
		for (size_t i = 0; i < 4; ++i)
		{
			const float		*srcPtr = src;
			const float		*mulPtrStop = mulPtr + mul.m_Columns;
			__m128			curAccum_xyzw = _mm_set1_ps(0.0f);

			mulPtrStop -= 4;
			while (mulPtr <= mulPtrStop)
			{
				const __m128	multiplier = _mm_load_ps(mulPtr);
				const __m128	value = _SrcAligned ? _mm_load_ps(srcPtr) : _mm_loadu_ps(srcPtr);
				const __m128	product = _mm_mul_ps(multiplier, value);
				curAccum_xyzw = _mm_add_ps(curAccum_xyzw, product);
				mulPtr += 4;
				srcPtr += 4;
			}
			mulPtrStop += 4;
			if (mulPtr < mulPtrStop)
			{
				ptrdiff_t		floatsLeft = mulPtrStop - mulPtr;
				const __m128	mask = simdLastValuesMask[floatsLeft - 1];
				const __m128	multiplier = _mm_and_ps(_mm_load_ps(mulPtr), mask);
				const __m128	value = _mm_and_ps(_SrcAligned ? _mm_load_ps(srcPtr) : _mm_loadu_ps(srcPtr), mask);
				const __m128	product = _mm_mul_ps(multiplier, value);

				curAccum_xyzw = _mm_add_ps(curAccum_xyzw, product);
				mulPtr += 4;
			}
			// Horizontal sum of accum:
			const __m128	accum_zwxy = _mm_shuffle_ps(curAccum_xyzw, curAccum_xyzw, _MM_SHUFFLE(1, 0, 3, 2));
			const __m128	reduc1_xyxy = _mm_add_ps(curAccum_xyzw, accum_zwxy);
			const __m128	reduc1_yxyx = _mm_shuffle_ps(reduc1_xyxy, reduc1_xyxy, _MM_SHUFFLE(0, 1, 0, 1));
			const __m128	reduc2_xxxx = _mm_add_ps(reduc1_yxyx, reduc1_xyxy);
			const __m128	keepOneMask = simdMasksKeepOne[i];

			accum_xyzw = _mm_add_ps(_mm_and_ps(reduc2_xxxx, keepOneMask), accum_xyzw);
		}

		const __m128	add_xyzw = _AddAligned ? _mm_load_ps(addPtr) : _mm_loadu_ps(addPtr);

		accum_xyzw = _mm_add_ps(accum_xyzw, add_xyzw);
		if (_DstAligned)
			_mm_store_ps(dstPtr, accum_xyzw);
		else
			_mm_storeu_ps(dstPtr, accum_xyzw);
		dstPtr += 4;
		addPtr += 4;
	}
	dstPtrStop += 4;
	while (dstPtr < dstPtrStop)
	{
		__m128			accum_xyzw = _mm_set1_ps(0.0f);
		const float		*srcPtr = src;
		const float		*mulPtrStop = mulPtr + mul.m_Columns;

		mulPtrStop -= 4;
		while (mulPtr <= mulPtrStop)
		{
			const __m128	multiplier = _mm_load_ps(mulPtr);
			const __m128	value = _SrcAligned ? _mm_load_ps(srcPtr) : _mm_loadu_ps(srcPtr);
			const __m128	product = _mm_mul_ps(multiplier, value);
			accum_xyzw = _mm_add_ps(accum_xyzw, product);
			mulPtr += 4;
			srcPtr += 4;
		}
		mulPtrStop += 4;
		if (mulPtr < mulPtrStop)
		{
			ptrdiff_t		floatsLeft = mulPtrStop - mulPtr;
			const __m128	mask = simdLastValuesMask[floatsLeft - 1];
			const __m128	multiplier = _mm_and_ps(_mm_load_ps(mulPtr), mask);
			const __m128	value = _mm_and_ps(_SrcAligned ? _mm_load_ps(srcPtr) : _mm_loadu_ps(srcPtr), mask);
			const __m128	product = _mm_mul_ps(multiplier, value);

			accum_xyzw = _mm_add_ps(accum_xyzw, product);
			mulPtr += 4;
		}
		// Horizontal sum of accum:
		const __m128	accum_zwxy = _mm_shuffle_ps(accum_xyzw, accum_xyzw, _MM_SHUFFLE(1, 0, 3, 2));
		const __m128	reduc1_xyxy = _mm_add_ps(accum_xyzw, accum_zwxy);
		const __m128	reduc1_yxyx = _mm_shuffle_ps(reduc1_xyxy, reduc1_xyxy, _MM_SHUFFLE(0, 1, 0, 1));
		const __m128	reduc2_xxxx = _mm_add_ps(reduc1_yxyx, reduc1_xyxy);

		*dstPtr = reduc2_xxxx.m128_f32[0] + *addPtr;
		dstPtr += 1;
		addPtr += 1;
	}
}

void	CNeuronMatrix::ComputeNetInput(float *dst, const float *src, const SConstNeuronMatrixView &mul, const float *add)
{
	bool			srcIsAligned = ((ptrdiff_t)src & 0xF) == 0;
	bool			dstIsAligned = ((ptrdiff_t)dst & 0xF) == 0;
	bool			addIsAligned = ((ptrdiff_t)add & 0xF) == 0;

	if (dstIsAligned && srcIsAligned && addIsAligned)
		_ComputeNetInput<true, true, true>(dst, src, mul, add);
	else if (dstIsAligned && srcIsAligned && !addIsAligned)
		_ComputeNetInput<true, true, false>(dst, src, mul, add);
	else if (dstIsAligned && !srcIsAligned && addIsAligned)
		_ComputeNetInput<true, false, true>(dst, src, mul, add);
	else if (dstIsAligned && !srcIsAligned && !addIsAligned)
		_ComputeNetInput<true, false, false>(dst, src, mul, add);
	else if (!dstIsAligned && srcIsAligned && addIsAligned)
		_ComputeNetInput<false, true, true>(dst, src, mul, add);
	else if (!dstIsAligned && srcIsAligned && !addIsAligned)
		_ComputeNetInput<false, true, false>(dst, src, mul, add);
	else if (!dstIsAligned && !srcIsAligned && addIsAligned)
		_ComputeNetInput<false, false, true>(dst, src, mul, add);
	else if (!dstIsAligned && !srcIsAligned && !addIsAligned)
		_ComputeNetInput<false, false, false>(dst, src, mul, add);
	else
		assert(false);
}

void	CNeuronMatrix::ComputeError(float *dstProd, const float *src, const SConstNeuronMatrixView &mul)
{
#if		0
	// Reference non-SIMD code
	for (size_t x = 0; x < mul.m_Columns; ++x)
	{
		float	dstValue = 0.0f;
		for (size_t y = 0; y < mul.m_Rows; ++y)
		{
			const float 	*mulData = mul.GetRow(y) + x;
			dstValue += src[y] * *mulData;
		}
		dstProd[x] = dstValue;
	}
#else
	const size_t	mulStride = mul.RowStride();
	float			*dstPtr = dstProd;
	float			*dstPtrStop = dstProd + mul.m_Columns;
	const float		*mulPtr = mul.m_Data;
	dstPtrStop -= 4;
	while (dstPtr <= dstPtrStop)
	{
		const float		*curMulPtr = mulPtr;
		const float		*srcPtr = src;
		const float		*srcPtrStop = src + mul.m_Rows;
		__m128			accum_xyzw = _mm_set1_ps(0.0f);

		srcPtrStop -= 4;
		while (srcPtr <= srcPtrStop)
		{
			const __m128	src_xyzw = _mm_loadu_ps(srcPtr);
			const __m128	src_xxxx = _mm_shuffle_ps(src_xyzw, src_xyzw, _MM_SHUFFLE(0, 0, 0, 0));
			const __m128	src_yyyy = _mm_shuffle_ps(src_xyzw, src_xyzw, _MM_SHUFFLE(1, 1, 1, 1));
			const __m128	src_zzzz = _mm_shuffle_ps(src_xyzw, src_xyzw, _MM_SHUFFLE(2, 2, 2, 2));
			const __m128	src_wwww = _mm_shuffle_ps(src_xyzw, src_xyzw, _MM_SHUFFLE(3, 3, 3, 3));

			const __m128	mul1_xyzw = _mm_loadu_ps(curMulPtr);
			const __m128	mul2_xyzw = _mm_loadu_ps(curMulPtr + mulStride);
			const __m128	mul3_xyzw = _mm_loadu_ps(curMulPtr + 2 * mulStride);
			const __m128	mul4_xyzw = _mm_loadu_ps(curMulPtr + 3 * mulStride);

			const __m128	mulsrc1_xyzw = _mm_mul_ps(src_xxxx, mul1_xyzw);
			const __m128	mulsrc2_xyzw = _mm_mul_ps(src_yyyy, mul2_xyzw);
			const __m128	mulsrc3_xyzw = _mm_mul_ps(src_zzzz, mul3_xyzw);
			const __m128	mulsrc4_xyzw = _mm_mul_ps(src_wwww, mul4_xyzw);

			const __m128	add1_xyzw = _mm_add_ps(mulsrc3_xyzw, mulsrc4_xyzw);
			const __m128	add2_xyzw = _mm_add_ps(mulsrc2_xyzw, add1_xyzw);
			const __m128	add3_xyzw = _mm_add_ps(mulsrc1_xyzw, add2_xyzw);

			accum_xyzw = _mm_add_ps(accum_xyzw, add3_xyzw);
			curMulPtr += 4 * mulStride;
			srcPtr += 4;
		}
		srcPtrStop += 4;
		if (srcPtr < srcPtrStop)
		{
			const ptrdiff_t	floatsLeft = srcPtrStop - srcPtr;
			const __m128	src_xyzw = _mm_loadu_ps(srcPtr);
			const __m128	src_xxxx = _mm_shuffle_ps(src_xyzw, src_xyzw, _MM_SHUFFLE(0, 0, 0, 0));
			const __m128	src_yyyy = _mm_shuffle_ps(src_xyzw, src_xyzw, _MM_SHUFFLE(1, 1, 1, 1));
			const __m128	src_zzzz = _mm_shuffle_ps(src_xyzw, src_xyzw, _MM_SHUFFLE(2, 2, 2, 2));
			const __m128	mul1_xyzw = _mm_loadu_ps(curMulPtr);

			if (floatsLeft == 1)
			{
				accum_xyzw = _mm_add_ps(accum_xyzw, _mm_mul_ps(src_xxxx, mul1_xyzw));
			}
			else if (floatsLeft == 2)
			{
				const __m128	mul2_xyzw = _mm_loadu_ps(curMulPtr + mulStride);
				accum_xyzw = _mm_add_ps(accum_xyzw, _mm_mul_ps(src_xxxx, mul1_xyzw));
				accum_xyzw = _mm_add_ps(accum_xyzw, _mm_mul_ps(src_yyyy, mul2_xyzw));
			}
			else if (floatsLeft == 3)
			{
				const __m128	mul2_xyzw = _mm_loadu_ps(curMulPtr + mulStride);
				const __m128	mul3_xyzw = _mm_loadu_ps(curMulPtr + 2 * mulStride);
				accum_xyzw = _mm_add_ps(accum_xyzw, _mm_mul_ps(src_xxxx, mul1_xyzw));
				accum_xyzw = _mm_add_ps(accum_xyzw, _mm_mul_ps(src_yyyy, mul2_xyzw));
				accum_xyzw = _mm_add_ps(accum_xyzw, _mm_mul_ps(src_zzzz, mul3_xyzw));
			}
			else
				assert(false);
		}
		_mm_storeu_ps(dstPtr, accum_xyzw);

		dstPtr += 4;
		mulPtr += 4;
	}
	dstPtrStop += 4;
	while (dstPtr < dstPtrStop)
	{
		const float		*srcPtr = src;
		const float		*srcPtrStop = srcPtr + mul.m_Rows;
		const float		*currentMulPtr = mulPtr;
		float			accum = 0.0f;

		while (srcPtr < srcPtrStop)
		{
			accum += *srcPtr * *currentMulPtr;
			currentMulPtr += mulStride;
			srcPtr += 1;
		}
		*dstPtr = accum;
		dstPtr += 1;
		mulPtr += 1;
	}
#endif
}
