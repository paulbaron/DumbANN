#pragma once

#include <stdlib.h>
#include <cstring>

struct	SNeuronMatrixView
{
	SNeuronMatrixView()
	:	m_Data(nullptr)
	,	m_RowByteStride(0)
	,	m_Rows(0)
	,	m_Columns(0)
	{
	}

	SNeuronMatrixView(float* data, size_t rows, size_t col, size_t rowStride);
	~SNeuronMatrixView();

	float		*GetRow(size_t idx) const;
	size_t		RowStride() const { return m_RowByteStride / sizeof(float); }

	float		*m_Data;
	size_t		m_RowByteStride;
	size_t		m_Rows;
	size_t		m_Columns;
};

struct	SConstNeuronMatrixView
{
	SConstNeuronMatrixView()
	:	m_Data(nullptr)
	,	m_RowByteStride(0)
	,	m_Rows(0)
	,	m_Columns(0)
	{
	}

	SConstNeuronMatrixView(const float *data, size_t rows, size_t col, size_t rowStride);
	SConstNeuronMatrixView(const SNeuronMatrixView &oth);
	~SConstNeuronMatrixView();

	const float	*GetRow(size_t idx) const;
	size_t		RowStride() const { return m_RowByteStride / sizeof(float); }

	const float	*m_Data;
	size_t		m_RowByteStride;
	size_t		m_Rows;
	size_t		m_Columns;
};

class	CNeuronVector
{
public:
	CNeuronVector();
	~CNeuronVector();

	bool	AllocateStorage(size_t elements);
	float	*Data() const { return m_Data; }
	size_t	Size() const { return m_Size; }

private:
	float	*m_Data;
	size_t	m_Size;
};

class	CNeuronMatrix
{
public:
	CNeuronMatrix();
	~CNeuronMatrix();

	bool	AllocMatrix(size_t rows, size_t col);

	float						*Data() const { return m_Mat.m_Data; }
	size_t						StorageByteSize() const { return m_Mat.m_RowByteStride * m_Mat.m_Rows; }
	const SNeuronMatrixView		&View() const { return m_Mat; }

	static void		ComputeNetInput(float *dst, const float *src, const SConstNeuronMatrixView &mul, const float *add);
	static void		ComputeError(float *dstProd, const float *src, const SConstNeuronMatrixView &mul);

private:
	SNeuronMatrixView	m_Mat;
};
