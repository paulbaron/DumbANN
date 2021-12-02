#pragma once

#include <stdlib.h>
#include <cstring>

struct	SNeuronMatrixView
{
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

struct	SConvolutionParams
{
public:
	size_t	m_KernelCount;
	size_t	m_KernelSizeX;
	size_t	m_KernelSizeY;
	size_t	m_KernelStride;
	size_t	m_InputPadding;
	size_t	m_InputImageCount;
	size_t	m_InputSizeX;
	size_t	m_InputSizeY;

	SConvolutionParams()
	:	m_KernelCount(0)
	,	m_KernelSizeX(0)
	,	m_KernelSizeY(0)
	,	m_KernelStride(0)
	,	m_InputPadding(0)
	,	m_InputImageCount(0)
	,	m_InputSizeX(0)
	,	m_InputSizeY(0)
	{
	}

	size_t	GetConvOutputSizeX() const
	{
		const size_t	convCount = (m_InputSizeX + 1 + 2 * m_InputPadding) - m_KernelSizeX;
		if (convCount % m_KernelStride != 0)
			return convCount / m_KernelStride + 1;
		return convCount / m_KernelStride;
	}

	size_t	GetConvOutputSizeY() const
	{
		const size_t	convCount = (m_InputSizeY + 1 + 2 * m_InputPadding) - m_KernelSizeY;
		if (convCount % m_KernelStride != 0)
			return convCount / m_KernelStride + 1;
		return convCount / m_KernelStride;
	}
};

template<typename T,
		void (T::*_Func)(	const float *,
							const SConvolutionParams &,
							size_t,
							int, int,
							int, int,
							int, int,
							int, int)>
void		Convolute(	T *cbClass,
						const float *input,
						size_t rangeMin, size_t rangeMax,
						const SConvolutionParams &convolution)
{
	const size_t	featureOutputSizeX = convolution.GetConvOutputSizeX();
	const size_t	featureOutputSizeY = convolution.GetConvOutputSizeY();
	const int		padding = static_cast<int>(convolution.m_InputPadding);

	// For each feature:
	for (size_t outFeatureIdx = rangeMin; outFeatureIdx < rangeMax; ++outFeatureIdx)
	{
		// Convolution:
		for (size_t convY = 0; convY < featureOutputSizeY; ++convY)
		{
			for (size_t convX = 0; convX < featureOutputSizeX; ++convX)
			{
				int				paddedConvX = (int)(convX * convolution.m_KernelStride) - padding;
				int				paddedConvY = (int)(convY * convolution.m_KernelStride) - padding;
				int				startInX = paddedConvX >= 0 ? 0 : -paddedConvX;
				int				startInY = paddedConvY >= 0 ? 0 : -paddedConvY;
				int				stopInX = convolution.m_KernelSizeX;
				int				stopInY = convolution.m_KernelSizeY;
				
				if (paddedConvX + convolution.m_KernelSizeX > convolution.m_InputSizeX)
					stopInX -= paddedConvX + convolution.m_KernelSizeX - convolution.m_InputSizeX;
				if (paddedConvY + convolution.m_KernelSizeY > convolution.m_InputSizeY)
					stopInY -= paddedConvY + convolution.m_KernelSizeY - convolution.m_InputSizeY;

				if (stopInX <= 0 || stopInY <= 0)
					continue;
				(cbClass->*_Func)(	input,
									convolution,
									outFeatureIdx,
									startInX, stopInX,
									startInY, stopInY,
									convX, convY,
									paddedConvX, paddedConvY);
			}
		}
	}
}

template<typename T,
		void (T::*_Func)(	float *,
							const SConvolutionParams &,
							size_t, size_t,
							int, int,
							int, int,
							int, int,
							int, int) const>
void	ConvoluteTranspose(	const T *cbClass,
							float *output,
							size_t rangeMin, size_t rangeMax,
							const SConvolutionParams &convolution)
{
	// Quite confusing: feature input is the "output" ptr
	// feature output writes into feature input... the naming is done for the regular convolution...
	const size_t	featureOutputSizeX = convolution.GetConvOutputSizeX();
	const size_t	featureOutputSizeY = convolution.GetConvOutputSizeY();
	const size_t	featureInputStride = convolution.m_InputSizeX * convolution.m_InputSizeY;
	const int		padding = static_cast<int>(convolution.m_InputPadding);

	memset(output + rangeMin * featureInputStride, 0, (rangeMax - rangeMin) * featureInputStride * sizeof(float));
	// For each input feature:
	for (size_t inFeatureIdx = rangeMin; inFeatureIdx < rangeMax; ++inFeatureIdx)
	{
		// For each output feature:
		for (size_t outFeatureIdx = 0; outFeatureIdx < convolution.m_KernelCount; ++outFeatureIdx)
		{
			// Convolution:
			for (size_t convY = 0; convY < featureOutputSizeY; ++convY)
			{
				for (size_t convX = 0; convX < featureOutputSizeX; ++convX)
				{
					int				paddedConvX = (int)(convX * convolution.m_KernelStride) - padding;
					int				paddedConvY = (int)(convY * convolution.m_KernelStride) - padding;
					int				startInX = (size_t)(paddedConvX >= 0 ? 0 : -paddedConvX);
					int				startInY = (size_t)(paddedConvY >= 0 ? 0 : -paddedConvY);
					int				stopInX = convolution.m_KernelSizeX;
					int				stopInY = convolution.m_KernelSizeY;
					
					if (paddedConvX + convolution.m_KernelSizeX > convolution.m_InputSizeX)
						stopInX -= paddedConvX + convolution.m_KernelSizeX - convolution.m_InputSizeX;
					if (paddedConvY + convolution.m_KernelSizeY > convolution.m_InputSizeY)
						stopInY -= paddedConvY + convolution.m_KernelSizeY - convolution.m_InputSizeY;
	
					if (stopInX <= 0 || stopInY <= 0)
						continue;
					(cbClass->*_Func)(	output,
										convolution,
										inFeatureIdx, outFeatureIdx,
										startInX, stopInX,
										startInY, stopInY,
										convX, convY,
										paddedConvX, paddedConvY);

				}
			}
		}
	}
}