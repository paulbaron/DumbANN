
#include "LayerConv2D.h"
#include <assert.h>

CLayerConv2D::CLayerConv2D()
{
}

CLayerConv2D::~CLayerConv2D()
{
}

bool	CLayerConv2D::Setup(size_t inputFeatureCount, size_t inputSizeX, size_t inputSizeY,
							size_t featureCount, size_t featureSizeX, size_t featureSizeY,
							size_t padding, size_t stride)
{
	if (stride == 0)
		stride = 1;
	if (inputSizeX % stride != 0 ||
		inputSizeY % stride != 0)
	{
		fprintf(stderr, "Input size should be a multiple of stride");
		assert(false);
		return false;
	}

	m_ConvParams.m_KernelCount = featureCount;
	m_ConvParams.m_KernelSizeX = featureSizeX;
	m_ConvParams.m_KernelSizeY = featureSizeY;
	m_ConvParams.m_KernelStride = stride;
	m_ConvParams.m_InputPadding = padding;
	m_ConvParams.m_InputImageCount = inputFeatureCount;
	m_ConvParams.m_InputSizeX = inputSizeX;
	m_ConvParams.m_InputSizeY = inputSizeY;
	
	const size_t	featureOutputSizeX = GetOutputSizeX();
	const size_t	featureOutputSizeY = GetOutputSizeY();
	assert(featureOutputSizeX != 0 && featureOutputSizeY != 0);
	const size_t	outputSize = featureCount * featureOutputSizeX * featureOutputSizeY;
	const size_t	weightsSizeX = inputFeatureCount * featureSizeX * featureSizeY;
	const size_t	weightsSizeY = featureCount;

	m_InputSize = inputFeatureCount * inputSizeX * inputSizeY;

	m_Weights.AllocMatrix(weightsSizeY, weightsSizeX);
	m_SlopesWeightAccum.AllocMatrix(weightsSizeY, weightsSizeX);
	m_SlopesOutAccum.AllocateStorage(outputSize);
	m_Bias.AllocateStorage(outputSize);

	// When using inertia we need those storages:
	m_DeltaWeightVelocity.AllocMatrix(weightsSizeY, weightsSizeX);
	m_DeltaBiasVelocity.AllocateStorage(outputSize);

	m_NetInput.AllocateStorage(outputSize);
	m_Output.AllocateStorage(outputSize);
	m_SlopesOut.AllocateStorage(outputSize);

	memset(m_SlopesWeightAccum.Data(), 0, m_SlopesWeightAccum.StorageByteSize());
	memset(m_SlopesOutAccum.Data(), 0, m_SlopesOutAccum.Size() * sizeof(float));

	// Initialize weights to random floats:
	Initializer();
	return true;
}

void	CLayerConv2D::FeedForward(const float *input, size_t rangeMin, size_t rangeMax)
{
	Convolute<CLayerConv2D, &CLayerConv2D::Kernel_ComputeNetInput>(this, input, rangeMin, rangeMax, m_ConvParams);
	const size_t	featureOutputSizeX = GetOutputSizeX();
	const size_t	featureOutputSizeY = GetOutputSizeY();
	const size_t	featureStide = featureOutputSizeX * featureOutputSizeY;
	const size_t	outputRange = (rangeMax - rangeMin) * featureStide;
	Activation(m_Output.Data() + featureStide * rangeMin, m_NetInput.Data() + featureStide * rangeMin, outputRange);
}

void	CLayerConv2D::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
	const size_t	featureOutputSizeX = GetOutputSizeX();
	const size_t	featureOutputSizeY = GetOutputSizeY();
	const size_t	featureStide = featureOutputSizeX * featureOutputSizeY;
	const size_t	outputRange = (rangeMax - rangeMin) * featureStide;
	float			*netInputPtr = m_NetInput.Data();

	// Inner layer of the neural network:
	for (size_t i = featureStide * rangeMin; i < featureStide * rangeMax; ++i)
		m_SlopesOut.Data()[i] = -error[i];
	ActivationDerivative(m_SlopesOut.Data() + featureStide * rangeMin, netInputPtr + featureStide * rangeMin, outputRange);
	Convolute<CLayerConv2D, &CLayerConv2D::Kernel_AccumWeightsAndBiasDerivative>(this, prevOutput, rangeMin, rangeMax, m_ConvParams);
//	AccumWeightsAndBiasDerivative(prevOutput, rangeMin, rangeMax);
}

void	CLayerConv2D::BackPropagateError(const float *prevOutput, const CLayer* nextLayer, size_t rangeMin, size_t rangeMax)
{
	const size_t	featureOutputSizeX = GetOutputSizeX();
	const size_t	featureOutputSizeY = GetOutputSizeY();
	const size_t	featureStide = featureOutputSizeX * featureOutputSizeY;
	const size_t	outputRange = (rangeMax - rangeMin) * featureStide;
	float			*netInputPtr = m_NetInput.Data();

	// Inner layer of the neural network:
	nextLayer->GatherSlopes(m_SlopesOut.Data(), featureStide * rangeMin, featureStide * rangeMax);
	ActivationDerivative(m_SlopesOut.Data() + featureStide * rangeMin, netInputPtr + featureStide * rangeMin, outputRange);
	Convolute<CLayerConv2D, &CLayerConv2D::Kernel_AccumWeightsAndBiasDerivative>(this, prevOutput, rangeMin, rangeMax, m_ConvParams);
//	AccumWeightsAndBiasDerivative(prevOutput, rangeMin, rangeMax);
}

void	CLayerConv2D::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
	const size_t	outputRange = rangeMax - rangeMin;
	float			*slopeAccumPtr = m_SlopesOutAccum.Data();
	float			*biasesPtr = m_Bias.Data();
	const size_t	featureOutputSizeX = GetOutputSizeX();
	const size_t	featureOutputSizeY = GetOutputSizeY();
	const size_t	featureOutputStride = featureOutputSizeX * featureOutputSizeY;

	OptimizeWeight(rangeMin, rangeMax, trainingSteps);
	OptimizeBias(biasesPtr, slopeAccumPtr, rangeMin * featureOutputStride, rangeMax * featureOutputStride, trainingSteps);

	memset(m_SlopesWeightAccum.View().GetRow(rangeMin), 0, outputRange * m_SlopesWeightAccum.View().m_RowByteStride);
	memset(m_SlopesOutAccum.Data() + rangeMin * featureOutputStride, 0, outputRange * featureOutputStride * sizeof(float));
}

void	CLayerConv2D::GatherSlopes(float *dst, size_t rangeMin, size_t rangeMax) const
{
	const size_t	featureInputStride = m_ConvParams.m_InputSizeX * m_ConvParams.m_InputSizeY;
	const size_t	dstRange = rangeMax - rangeMin;
	ConvoluteTranspose<CLayerConv2D, &CLayerConv2D::Kernel_GatherSlopes>(	this,
																			dst,
																			rangeMin / featureInputStride,
																			rangeMax / featureInputStride,
																			m_ConvParams);
}

size_t	CLayerConv2D::GetThreadingHint() const
{
	size_t test1 = GetOutputSizeX();
	size_t test2 = GetOutputSizeY();
	return m_Weights.View().m_Columns * m_Weights.View().m_Rows * GetOutputSizeX() * GetOutputSizeY();
}

size_t	CLayerConv2D::GetDomainSize() const
{
	return m_ConvParams.m_KernelCount;
}

// This is going called for each convolution
// Be careful !

void	CLayerConv2D::Kernel_AccumWeightsAndBiasDerivative(	const float *input,
															const SConvolutionParams &conv,
															size_t featureIdx,
															int startInX, int stopInX,
															int startInY, int stopInY,
															int convX, int convY,
															int paddedConvX, int paddedConvY)
{
	const size_t	featureInputStride = conv.m_InputSizeX * conv.m_InputSizeY;
	const size_t	outFeatureWeightStride = conv.m_KernelSizeX * conv.m_KernelSizeY;
	const size_t	featureOutputSizeX = conv.GetConvOutputSizeX();
	const size_t	featureOutputSizeY = conv.GetConvOutputSizeY();
	float			*weightAccumPtr = m_SlopesWeightAccum.View().GetRow(featureIdx);
	const float		*slopePtr = m_SlopesOut.Data();
	float			*slopeAccumPtr = m_SlopesOutAccum.Data();
	const size_t	outIdx =	featureIdx * featureOutputSizeX * featureOutputSizeY +
								convY * featureOutputSizeX +
								convX;
	const float		slope = slopePtr[outIdx];

	// For each input feature:
	for (size_t inFeatureIdx = 0; inFeatureIdx < m_ConvParams.m_InputImageCount; ++inFeatureIdx)
	{
		for (size_t inY = startInY; inY < stopInY; ++inY)
		{
			for (size_t inX = startInX; inX < stopInX; ++inX)
			{
				size_t		inputIdx =	inFeatureIdx * featureInputStride +
										(inY + paddedConvY) * m_ConvParams.m_InputSizeX +
										(inX + paddedConvX);
				size_t		weightIdx = inFeatureIdx * outFeatureWeightStride + 
										inY * m_ConvParams.m_KernelSizeX + inX;
				weightAccumPtr[weightIdx] += input[inputIdx] * slope;
			}
		}
	}
}

void	CLayerConv2D::Kernel_ComputeNetInput(	const float *input,
												const SConvolutionParams &conv,
												size_t featureIdx,
												int startInX, int stopInX,
												int startInY, int stopInY,
												int convX, int convY,
												int paddedConvX, int paddedConvY)
{
	const size_t	featureInputStride = conv.m_InputSizeX * conv.m_InputSizeY;
	const size_t	outFeatureWeightStride = conv.m_KernelSizeX * conv.m_KernelSizeY;
	const size_t	featureOutputSizeX = conv.GetConvOutputSizeX();
	const size_t	featureOutputSizeY = conv.GetConvOutputSizeY();
	const float		*weightsPtr = m_Weights.View().GetRow(featureIdx);

	float		accum = 0.0f;
	// For each input feature:
	for (size_t inFeatureIdx = 0; inFeatureIdx < conv.m_InputImageCount; ++inFeatureIdx)
	{
		for (size_t inY = startInY; inY < stopInY; ++inY)
		{
			for (size_t inX = startInX; inX < stopInX; ++inX)
			{
				size_t	inputIdx =	inFeatureIdx * featureInputStride +
									(inY + paddedConvY) * conv.m_InputSizeX +
									(inX + paddedConvX);
				size_t	weightIdx = inFeatureIdx * outFeatureWeightStride + 
									inY * conv.m_KernelSizeX + inX;
				accum += input[inputIdx] * weightsPtr[weightIdx];
			}
		}
	}
	const size_t	outIdx =	featureIdx * featureOutputSizeX * featureOutputSizeY +
								convY * featureOutputSizeX +
								convX;
	accum += m_Bias.Data()[outIdx];
	m_NetInput.Data()[outIdx] = accum;
}

void	CLayerConv2D::Kernel_GatherSlopes(	float *output,
											const SConvolutionParams &conv,
											size_t inFeatureIdx,
											size_t outFeatureIdx,
											int startInX, int stopInX,
											int startInY, int stopInY,
											int convX, int convY,
											int paddedConvX, int paddedConvY) const
{
	const size_t	featureInputStride = conv.m_InputSizeX * conv.m_InputSizeY;
	const size_t	outFeatureWeightStride = conv.m_KernelSizeX * conv.m_KernelSizeY;
	const size_t	featureOutputSizeX = conv.GetConvOutputSizeX();
	const size_t	featureOutputSizeY = conv.GetConvOutputSizeY();
	float			*weightsPtr = m_Weights.View().GetRow(outFeatureIdx);
	const float		*slopePtr = m_SlopesOut.Data();
	const size_t	outIdx =	outFeatureIdx * featureOutputSizeX * featureOutputSizeY +
								convY * featureOutputSizeX +
								convX;
	const float		slope = slopePtr[outIdx];

	for (size_t inY = startInY; inY < stopInY; ++inY)
	{
		for (size_t inX = startInX; inX < stopInX; ++inX)
		{
			size_t	dstIdx =	inFeatureIdx * featureInputStride +
								(inY + paddedConvY) * conv.m_InputSizeX +
								(inX + paddedConvX);
			size_t	weightIdx = inFeatureIdx * outFeatureWeightStride + 
								inY * conv.m_KernelSizeX + inX;
			output[dstIdx] += slope * weightsPtr[weightIdx];
		}
	}
}

