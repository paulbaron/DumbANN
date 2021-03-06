
#include "LayerConv2D.h"
#include "NeuronKernel.h"
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

	m_KernelCount = featureCount;
	m_InputImageCount = inputFeatureCount;

	m_ConvParams.m_KernelSizeX = featureSizeX;
	m_ConvParams.m_KernelSizeY = featureSizeY;
	m_ConvParams.m_KernelStride = stride;
	m_ConvParams.m_InputPadding = padding;
	m_ConvParams.m_InputSizeX = inputSizeX;
	m_ConvParams.m_InputSizeY = inputSizeY;
	m_ConvParams.ComputeConvOutputSize();

	const size_t	featureOutputSizeX = GetOutputSizeX();
	const size_t	featureOutputSizeY = GetOutputSizeY();
	assert(featureOutputSizeX != 0 && featureOutputSizeY != 0);
	const size_t	outputSize = featureCount * featureOutputSizeX * featureOutputSizeY;
	const size_t	weightsSizeX = inputFeatureCount * featureSizeX * featureSizeY;
	const size_t	weightsSizeY = featureCount;

	m_InputSize = inputFeatureCount * inputSizeX * inputSizeY;
	m_OutputSize = outputSize;

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

	m_AdagradWeightAccum.AllocMatrix(weightsSizeY, weightsSizeX);
	m_AdagradBiasAccum.AllocateStorage(outputSize);

	for (size_t y = 0; y < m_AdagradWeightAccum.View().m_Rows; ++y)
	{
		float	*weightAccum = m_AdagradWeightAccum.View().GetRow(y);
		for (size_t x = 0; x < m_AdagradWeightAccum.View().m_Columns; ++x)
		{
			weightAccum[x] = 1.0f;
		}
	}
	for (size_t x = 0; x < m_AdagradBiasAccum.Size(); ++x)
	{
		m_AdagradBiasAccum.Data()[x] = 1.0f;
	}

	// Initialize weights to random floats:
	Initializer();
	return true;
}

void	CLayerConv2D::FeedForward(const float *input, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerConv2D", "CLayerConv2D::FeedForward", MP_GREEN1);
	SComputeNetInput_KernelIn	kernelIn;

	kernelIn.m_Bias = m_Bias.Data();
	kernelIn.m_InFeatureCount = m_InputImageCount;
	kernelIn.m_Input = input;
	kernelIn.m_NetInput = m_NetInput.Data();
	kernelIn.m_OutFeatureCount = m_KernelCount;
	kernelIn.m_Weights = m_Weights.View();

	KernelConvolute<SComputeNetInput_KernelIn,
					&CLayerConv2D::Kernel_ComputeNetInput>(kernelIn, rangeMin, rangeMax, m_ConvParams);

	const size_t	featureStride = m_ConvParams.m_OutputSizeX * m_ConvParams.m_OutputSizeY;
	const size_t	outputRange = (rangeMax - rangeMin) * featureStride;
	Activation(m_Output.Data() + featureStride * rangeMin, m_NetInput.Data() + featureStride * rangeMin, outputRange);
}

void	CLayerConv2D::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerConv2D", "CLayerConv2D::BackPropagateError", MP_RED1);
	const size_t	featureStride = m_ConvParams.m_OutputSizeX * m_ConvParams.m_OutputSizeY;
	const size_t	outputRange = (rangeMax - rangeMin) * featureStride;
	float			*netInputPtr = m_NetInput.Data();

	// Outter layer of the neural network:
	for (size_t i = featureStride * rangeMin; i < featureStride * rangeMax; ++i)
		m_SlopesOut.Data()[i] = -error[i];
	ActivationDerivative(m_SlopesOut.Data() + featureStride * rangeMin, netInputPtr + featureStride * rangeMin, outputRange);

	if (m_Learn)
	{
		SAccumWeightsAndBiasDerivative_KernelIn	kernelIn;
	
		kernelIn.m_InFeatureCount = m_InputImageCount;
		kernelIn.m_OutFeatureCount = m_KernelCount;
		kernelIn.m_Input = prevOutput;
		kernelIn.m_AccumBias = m_SlopesOutAccum.Data();
		kernelIn.m_AccumWeights = m_SlopesWeightAccum.View();
		kernelIn.m_Slopes = m_SlopesOut.Data();
	
		KernelConvolute<SAccumWeightsAndBiasDerivative_KernelIn,
						&CLayerConv2D::Kernel_AccumWeightsAndBiasDerivative>(kernelIn, rangeMin, rangeMax, m_ConvParams);
	}
}

void	CLayerConv2D::BackPropagateError(const float *prevOutput, const CLayer* nextLayer, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerConv2D", "CLayerConv2D::BackPropagateError", MP_RED1);
	const size_t	featureStride = m_ConvParams.m_OutputSizeX * m_ConvParams.m_OutputSizeY;
	const size_t	outputRange = (rangeMax - rangeMin) * featureStride;
	float			*netInputPtr = m_NetInput.Data();

	// Inner layer of the neural network:
	ActivationDerivative(m_SlopesOut.Data() + featureStride * rangeMin, netInputPtr + featureStride * rangeMin, outputRange);

	if (m_Learn)
	{
		SAccumWeightsAndBiasDerivative_KernelIn	kernelIn;
	
		kernelIn.m_InFeatureCount = m_InputImageCount;
		kernelIn.m_OutFeatureCount = m_KernelCount;
		kernelIn.m_Input = prevOutput;
		kernelIn.m_AccumBias = m_SlopesOutAccum.Data();
		kernelIn.m_AccumWeights = m_SlopesWeightAccum.View();
		kernelIn.m_Slopes = m_SlopesOut.Data();
	
		KernelConvolute<SAccumWeightsAndBiasDerivative_KernelIn,
						&CLayerConv2D::Kernel_AccumWeightsAndBiasDerivative>(kernelIn, rangeMin, rangeMax, m_ConvParams);
	}
}

void	CLayerConv2D::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerConv2D", "CLayerConv2D::UpdateWeightsAndBias", MP_BLUE1);
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

void	CLayerConv2D::GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const
{
	MICROPROFILE_SCOPEI("CLayerConv2D", "CLayerConv2D::GatherSlopes", MP_PALEVIOLETRED1);
	(void)prevLayer;
	const size_t	featureInputStride = m_ConvParams.m_InputSizeX * m_ConvParams.m_InputSizeY;
	const size_t	dstRange = rangeMax - rangeMin;

	memset(dst + rangeMin, 0, (rangeMax - rangeMin) * sizeof(float));

	SGatherSlopes_KernelIn	kernelIn;

	kernelIn.m_InFeatureCount = m_InputImageCount;
	kernelIn.m_OutFeatureCount = m_KernelCount;
	kernelIn.m_Slopes = m_SlopesOut.Data();
	kernelIn.m_Weights = m_Weights.View();
	kernelIn.m_Output = dst;

	KernelConvolute<SGatherSlopes_KernelIn,
					&CLayerConv2D::Kernel_GatherSlopes>(kernelIn,
														rangeMin / featureInputStride,
														rangeMax / featureInputStride,
														m_ConvParams);
}

void	CLayerConv2D::PrintInfo() const
{
	printf("\tLayer Convolution 2D:\n");
	printf(	"\t\tInput: %zu %zux%zu (padding: %zu)\n",
			m_InputImageCount, m_ConvParams.m_InputSizeX, m_ConvParams.m_InputSizeY,
			m_ConvParams.m_InputPadding);
	printf(	"\t\tFeatures: %zu %zux%zu (stride: %zu)\n",
			m_KernelCount, m_ConvParams.m_KernelSizeX, m_ConvParams.m_KernelSizeY,
			m_ConvParams.m_KernelStride);
	printf("\t\tOutput: %zu %zux%zu\n", m_KernelCount, m_ConvParams.m_OutputSizeX, m_ConvParams.m_OutputSizeY);
	PrintBasicInfo();
}

void	CLayerConv2D::Serialize(std::vector<uint8_t> &data) const
{
	SerializeLayerType(data, ELayerType::LayerConv2D);
	SerializeInOutSize(data);
	m_ConvParams.Serialize(data);
	size_t		prevSize = data.size();
	data.resize(prevSize + 2 * sizeof(uint32_t));
	uint32_t	*dataPtr = (uint32_t*)(data.data() + prevSize);
	dataPtr[0] = m_KernelCount;
	dataPtr[1] = m_InputImageCount;
	SerializeWeightsAndBias(data);
}

bool	CLayerConv2D::UnSerialize(const std::vector<uint8_t> &data, size_t &curIdx)
{
	if (!UnSerializeInOutSize(data, curIdx))
		return false;
	if (!m_ConvParams.UnSerialize(data, curIdx))
		return false;
	if (curIdx + 2 * sizeof(uint32_t) > data.size())
		return false;
	uint32_t	*dataPtr = (uint32_t*)(data.data() + curIdx);
	m_KernelCount = dataPtr[0];
	m_InputImageCount = dataPtr[1];
	curIdx += 2 * sizeof(uint32_t);
	if (!Setup(	m_InputImageCount, m_ConvParams.m_InputSizeX, m_ConvParams.m_InputSizeY,
				m_KernelCount, m_ConvParams.m_KernelSizeX, m_ConvParams.m_KernelSizeY,
				m_ConvParams.m_InputPadding, m_ConvParams.m_KernelStride))
		return false;
	if (!UnSerializeWeightsAndBias(data, curIdx))
		return false;
	return true;
}

size_t	CLayerConv2D::GetThreadingHint() const
{
	return GetOutputSizeX() * GetOutputSizeY() * m_KernelCount * 4096;
}

size_t	CLayerConv2D::GetDomainSize() const
{
	return m_KernelCount;
}

// This is going called for each convolution
// Be careful !

void	CLayerConv2D::Kernel_AccumWeightsAndBiasDerivative(	const SAccumWeightsAndBiasDerivative_KernelIn &input,
															const SKernelRange &range,
															const SConvolutionParams &conv)
{
	const size_t	featureInputStride = conv.m_InputSizeX * conv.m_InputSizeY;
	const size_t	featureOutputStride = conv.m_OutputSizeX * conv.m_OutputSizeY;
	const size_t	outFeatureWeightStride = conv.m_KernelSizeX * conv.m_KernelSizeY;
	float			*weightsAccumPtr = input.m_AccumWeights.GetRow(range.m_FeatureIdx);
	const size_t	outIdx =	range.m_FeatureIdx * featureOutputStride +
								range.m_ConvIdxY * conv.m_OutputSizeX +
								range.m_ConvIdxX;
	const float		slope = input.m_Slopes[outIdx];

	assert(abs(slope) < 1000000.0f);
	assert(!isnan(slope));
	assert(!isinf(slope));
	input.m_AccumBias[outIdx] += slope;
	assert(abs(input.m_AccumBias[outIdx]) < 1000000.0f);
	assert(!isnan(input.m_AccumBias[outIdx]));
	assert(!isinf(input.m_AccumBias[outIdx]));
	// For each input feature:
	for (size_t inFeatureIdx = 0; inFeatureIdx < input.m_InFeatureCount; ++inFeatureIdx)
	{
		for (size_t inY = range.m_StartConvY; inY < range.m_StopConvY; ++inY)
		{
			for (size_t inX = range.m_StartConvX; inX < range.m_StopConvX; ++inX)
			{
				size_t	inputIdx =	inFeatureIdx * featureInputStride +
									inY * conv.m_InputSizeX +
									inX;
				size_t	weightIdx = inFeatureIdx * outFeatureWeightStride + 
									(inY - range.m_ConvOffsetY) * conv.m_KernelSizeX +
									(inX - range.m_ConvOffsetX);
				weightsAccumPtr[weightIdx] += input.m_Input[inputIdx] * slope;
				assert(abs(weightsAccumPtr[weightIdx]) < 1000000.0f);
				assert(!isnan(weightsAccumPtr[weightIdx]));
				assert(!isinf(weightsAccumPtr[weightIdx]));
			}
		}
	}
}

void	CLayerConv2D::Kernel_ComputeNetInput(	const SComputeNetInput_KernelIn &input,
												const SKernelRange &range,
												const SConvolutionParams &conv)
{
	const size_t	featureInputStride = conv.m_InputSizeX * conv.m_InputSizeY;
	const size_t	featureOutputStride = conv.m_OutputSizeX * conv.m_OutputSizeY;
	const size_t	outFeatureWeightStride = conv.m_KernelSizeX * conv.m_KernelSizeY;
	const float		*weightsPtr = input.m_Weights.GetRow(range.m_FeatureIdx);

	float		accum = 0.0f;
	// For each input feature:
	for (size_t inFeatureIdx = 0; inFeatureIdx < input.m_InFeatureCount; ++inFeatureIdx)
	{
		for (size_t inY = range.m_StartConvY; inY < range.m_StopConvY; ++inY)
		{
			for (size_t inX = range.m_StartConvX; inX < range.m_StopConvX; ++inX)
			{
				size_t	inputIdx =	inFeatureIdx * featureInputStride +
									inY * conv.m_InputSizeX +
									inX;
				size_t	weightIdx = inFeatureIdx * outFeatureWeightStride + 
									(inY - range.m_ConvOffsetY) * conv.m_KernelSizeX +
									(inX - range.m_ConvOffsetX);
				accum += input.m_Input[inputIdx] * weightsPtr[weightIdx];
				assert(abs(accum) < 1000000.0f);
				assert(!isnan(accum));
				assert(!isinf(accum));
			}
		}
	}
	const size_t	outIdx =	range.m_FeatureIdx * featureOutputStride +
								range.m_ConvIdxY * conv.m_OutputSizeX +
								range.m_ConvIdxX;
	accum += input.m_Bias[outIdx];
	input.m_NetInput[outIdx] = accum;
	assert(abs(input.m_NetInput[outIdx]) < 1000000.0f);
	assert(!isnan(input.m_NetInput[outIdx]));
	assert(!isinf(input.m_NetInput[outIdx]));
}

void	CLayerConv2D::Kernel_GatherSlopes(	const SGatherSlopes_KernelIn &input,
											const SKernelRange &range,
											const SConvolutionParams &conv)
{

	const size_t	featureInputStride = conv.m_InputSizeX * conv.m_InputSizeY;
	const size_t	featureOutputStride = conv.m_OutputSizeX * conv.m_OutputSizeY;
	const size_t	outFeatureWeightStride = conv.m_KernelSizeX * conv.m_KernelSizeY;

	// For each output feature:
	for (size_t outFeatureIdx = 0; outFeatureIdx < input.m_OutFeatureCount; ++outFeatureIdx)
	{
		const size_t	outIdx =	outFeatureIdx * featureOutputStride +
									range.m_ConvIdxY * conv.m_OutputSizeX +
									range.m_ConvIdxX;
		const float		slope = input.m_Slopes[outIdx];
		float			*weightsPtr = input.m_Weights.GetRow(outFeatureIdx);

		for (size_t inY = range.m_StartConvY; inY < range.m_StopConvY; ++inY)
		{
			for (size_t inX = range.m_StartConvX; inX < range.m_StopConvX; ++inX)
			{
				size_t	dstIdx =	range.m_FeatureIdx * featureInputStride +
									inY * conv.m_InputSizeX +
									inX;
				size_t	weightIdx = range.m_FeatureIdx * outFeatureWeightStride + 
									(inY - range.m_ConvOffsetY) * conv.m_KernelSizeX +
									(inX - range.m_ConvOffsetX);
				input.m_Output[dstIdx] += slope * weightsPtr[weightIdx];
				assert(abs(input.m_Output[dstIdx]) < 100000000.0f);
				assert(!isnan(input.m_Output[dstIdx]));
				assert(!isinf(input.m_Output[dstIdx]));
			}
		}
	}
}

