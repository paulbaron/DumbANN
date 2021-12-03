#pragma once

#include <stdlib.h>
#include <cstring>
#include <assert.h>

struct	SConvolutionParams
{
public:
	size_t	m_KernelSizeX;
	size_t	m_KernelSizeY;
	size_t	m_KernelStride;
	size_t	m_InputPadding;
	size_t	m_InputSizeX;
	size_t	m_InputSizeY;
	size_t	m_OutputSizeX;
	size_t	m_OutputSizeY;

	SConvolutionParams()
	:	m_KernelSizeX(0)
	,	m_KernelSizeY(0)
	,	m_KernelStride(0)
	,	m_InputPadding(0)
	,	m_InputSizeX(0)
	,	m_InputSizeY(0)
	{
	}

	void	ComputeConvOutputSize()
	{
		const size_t	convCountX = (m_InputSizeX + 1 + 2 * m_InputPadding) - m_KernelSizeX;
		const size_t	convCountY = (m_InputSizeY + 1 + 2 * m_InputPadding) - m_KernelSizeY;

		if (convCountX % m_KernelStride != 0)
			m_OutputSizeX = convCountX / m_KernelStride + 1;
		else
			m_OutputSizeX = convCountX / m_KernelStride;

		if (convCountY % m_KernelStride != 0)
			m_OutputSizeY = convCountY / m_KernelStride + 1;
		else
			m_OutputSizeY = convCountY / m_KernelStride;
	}
};

struct	SKernelRange
{
	// Convolution idx:
	// m_ConvIdxX is from 0 -> m_OutputSizeX
	// m_ConvIdxY is from 0 -> m_OutputSizeY
	size_t	m_ConvIdxX;
	size_t	m_ConvIdxY;
	// Theoretical kernel offset:
	// (same as m_StartConvX and m_StartConvY with no padding)
	int		m_ConvOffsetX;
	int		m_ConvOffsetY;
	// Conv offset and size in input map:
	size_t	m_StartConvX;
	size_t	m_StartConvY;
	size_t	m_StopConvX;
	size_t	m_StopConvY;
	// Current feature index:
	// Input feature index in the case of forward propagation
	// Ouput feature index in the case of backward propagation or deconvolutional layers
	size_t	m_FeatureIdx;

	struct SKernelRange()
	:	m_ConvIdxX(0)
	,	m_ConvIdxY(0)
	,	m_ConvOffsetX(0)
	,	m_ConvOffsetY(0)
	,	m_StartConvX(0)
	,	m_StartConvY(0)
	,	m_StopConvX(0)
	,	m_StopConvY(0)
	,	m_FeatureIdx(0)
	{
	}
};

template<class _KernelIn, void (*_Kernel)(	const _KernelIn &,
											const SKernelRange &,
											const SConvolutionParams &)>
void		KernelConvolute(	const _KernelIn &kernelInput,
								size_t rangeMin, size_t rangeMax,
								const SConvolutionParams &convolution)
{
	const int		padding = static_cast<int>(convolution.m_InputPadding);
	SKernelRange	kernelRange;

	// For each feature:
	for (	kernelRange.m_FeatureIdx = rangeMin;
			kernelRange.m_FeatureIdx < rangeMax;
			++kernelRange.m_FeatureIdx)
	{
		// For each convolution:
		for (	kernelRange.m_ConvIdxY = 0;
				kernelRange.m_ConvIdxY < convolution.m_OutputSizeY;
				++kernelRange.m_ConvIdxY)
		{
			kernelRange.m_ConvOffsetY = (int)(kernelRange.m_ConvIdxY * convolution.m_KernelStride) - padding;
			kernelRange.m_StartConvY = std::max(0, kernelRange.m_ConvOffsetY);
			kernelRange.m_StopConvY = std::min(	kernelRange.m_ConvOffsetY + convolution.m_KernelSizeY,
												convolution.m_InputSizeY);
			
			assert(kernelRange.m_StartConvY < kernelRange.m_StopConvY);
			if (kernelRange.m_StartConvY < kernelRange.m_StopConvY)
			{
				for (	kernelRange.m_ConvIdxX = 0;
						kernelRange.m_ConvIdxX < convolution.m_OutputSizeX;
						++kernelRange.m_ConvIdxX)
				{
					kernelRange.m_ConvOffsetX = (int)(kernelRange.m_ConvIdxX * convolution.m_KernelStride) - padding;
					kernelRange.m_StartConvX = std::max(0, kernelRange.m_ConvOffsetX);
					kernelRange.m_StopConvX = std::min(kernelRange.m_ConvOffsetX + convolution.m_KernelSizeX, convolution.m_InputSizeX);
	
					assert(kernelRange.m_StartConvX < kernelRange.m_StopConvX);
					if (kernelRange.m_StartConvX < kernelRange.m_StopConvX)
					{
						_Kernel(kernelInput, kernelRange, convolution);
					}
				}
			}
		}
	}
}

#if		0
template<typename T,
		void (T::*_Func)(	float *,
							float *,
							const SConvolutionParams &,
							size_t, size_t,
							int, int,
							int, int,
							int, int,
							int, int) const>
void	ConvoluteTranspose(	const T *cbClass,
							float *output,
							float *input,
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
#endif