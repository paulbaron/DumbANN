
#include "LayerMaxPooling.h"
#include <assert.h>

CLayerMaxPooling2D::CLayerMaxPooling2D()
{
}

CLayerMaxPooling2D::~CLayerMaxPooling2D()
{
}

bool	CLayerMaxPooling2D::Setup(	size_t inputFeatureCount, size_t inputSizeX, size_t inputSizeY,
									size_t poolSizeX, size_t poolSizeY,
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

	m_FeatureCount = inputFeatureCount;

	m_ConvParams.m_KernelSizeX = poolSizeX;
	m_ConvParams.m_KernelSizeY = poolSizeY;
	m_ConvParams.m_KernelStride = stride;
	m_ConvParams.m_InputPadding = padding;
	m_ConvParams.m_InputSizeX = inputSizeX;
	m_ConvParams.m_InputSizeY = inputSizeY;
	m_ConvParams.ComputeConvOutputSize();

	m_OutputSize = m_FeatureCount * m_ConvParams.m_OutputSizeX * m_ConvParams.m_OutputSizeY;
	m_InputSize = inputFeatureCount * inputSizeX * inputSizeY;
	bool	success = true;
	success &= m_Output.AllocateStorage(m_OutputSize);
	success &= m_SlopesOut.AllocateStorage(m_OutputSize);
	return success;
}

void	CLayerMaxPooling2D::FeedForward(const float *input, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerMaxPooling2D", "CLayerMaxPooling2D::FeedForward", MP_GREEN1);
	SComputeOutput_KernelIn	kernelIn;

	kernelIn.m_FeatureCount = m_FeatureCount;
	kernelIn.m_Output = m_Output.Data();
	kernelIn.m_Input = input;

	const size_t	featureStride = m_ConvParams.m_OutputSizeX * m_ConvParams.m_OutputSizeY;
	KernelConvolute<SComputeOutput_KernelIn,
					&CLayerMaxPooling2D::Kernel_ComputeOutput>(kernelIn, rangeMin, rangeMax, m_ConvParams);
}

void	CLayerMaxPooling2D::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerMaxPooling2D", "CLayerMaxPooling2D::BackPropagateError", MP_RED1);
	const size_t	featureStide = m_ConvParams.m_OutputSizeX * m_ConvParams.m_OutputSizeY;

	// Outter layer of the neural network:
	for (size_t i = featureStide * rangeMin; i < featureStide * rangeMax; ++i)
		m_SlopesOut.Data()[i] = -error[i];
}

void	CLayerMaxPooling2D::BackPropagateError(const float *prevOutput, const CLayer* nextLayer, size_t rangeMin, size_t rangeMax)
{
}

void	CLayerMaxPooling2D::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
}

void	CLayerMaxPooling2D::GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const
{
	MICROPROFILE_SCOPEI("CLayerMaxPooling2D", "CLayerMaxPooling2D::GatherSlopes", MP_PALEVIOLETRED1);
	if (prevLayer == nullptr)
	{
		assert(false); // CLayerMaxPooling2D cannot be first layer
		return;
	}
	const size_t			featureInputStride = m_ConvParams.m_InputSizeX * m_ConvParams.m_InputSizeY;
	SGatherSlopes_KernelIn	kernelIn;

	kernelIn.m_FeatureCount = m_FeatureCount;
	kernelIn.m_Output = dst;
	kernelIn.m_Input = prevLayer->GetOutput().Data();
	kernelIn.m_Slopes = m_SlopesOut.Data();

	memset(dst + rangeMin, 0, (rangeMax - rangeMin) * sizeof(float));
	KernelConvolute<SGatherSlopes_KernelIn,
					&CLayerMaxPooling2D::Kernel_GatherSlopes>(kernelIn, rangeMin / featureInputStride, rangeMax / featureInputStride, m_ConvParams);
}

void	CLayerMaxPooling2D::PrintInfo() const
{
	printf("\tLayer Max Pooling 2D:\n");
	printf("\t\tInput: %zu %zux%zu (padding: %zu)\n",
			m_FeatureCount, m_ConvParams.m_InputSizeX, m_ConvParams.m_InputSizeY,
			m_ConvParams.m_InputPadding);
	printf(	"\t\tPool size: %zux%zu (stride: %zu)\n",
			m_ConvParams.m_KernelSizeX, m_ConvParams.m_KernelSizeY,
			m_ConvParams.m_KernelStride);
	printf("\t\tOutput: %zu %zux%zu\n",
			m_FeatureCount, m_ConvParams.m_OutputSizeX, m_ConvParams.m_OutputSizeY);
}

void	CLayerMaxPooling2D::Serialize(std::vector<uint8_t> &data) const
{
	SerializeLayerType(data, ELayerType::LayerMaxPooling);
	SerializeInOutSize(data);
	m_ConvParams.Serialize(data);
	size_t		prevSize = data.size();
	data.resize(prevSize + sizeof(uint32_t));
	uint32_t	*dataPtr = (uint32_t*)(data.data() + prevSize);
	dataPtr[0] = m_FeatureCount;
}

bool	CLayerMaxPooling2D::UnSerialize(const std::vector<uint8_t> &data, size_t &curIdx)
{
	if (!UnSerializeInOutSize(data, curIdx))
		return false;
	if (!m_ConvParams.UnSerialize(data, curIdx))
		return false;
	uint32_t	*dataPtr = (uint32_t*)(data.data() + curIdx);
	m_FeatureCount = dataPtr[0];
	curIdx += sizeof(uint32_t);
	if (!Setup(	m_FeatureCount, m_ConvParams.m_InputSizeX, m_ConvParams.m_InputSizeY,
				m_ConvParams.m_KernelSizeX, m_ConvParams.m_KernelSizeY,
				m_ConvParams.m_InputPadding, m_ConvParams.m_KernelStride))
		return false;
	return true;
}

size_t	CLayerMaxPooling2D::GetThreadingHint() const
{
	return m_FeatureCount * GetOutputSizeX() * GetOutputSizeY();
}

size_t	CLayerMaxPooling2D::GetDomainSize() const
{
	return m_FeatureCount;
}

void	CLayerMaxPooling2D::Kernel_ComputeOutput(	const SComputeOutput_KernelIn &input,
													const SKernelRange &range,
													const SConvolutionParams &conv)
{
	const size_t	featureInputStride = conv.m_InputSizeX * conv.m_InputSizeY;
	const size_t	featureOutputStride = conv.m_OutputSizeX * conv.m_OutputSizeY;
	const size_t	outFeatureWeightStride = conv.m_KernelSizeX * conv.m_KernelSizeY;
	float			maxValue = -FLT_MAX;

	for (size_t inY = range.m_StartConvY; inY < range.m_StopConvY; ++inY)
	{
		for (size_t inX = range.m_StartConvX; inX < range.m_StopConvX; ++inX)
		{
			size_t	inputIdx =	range.m_FeatureIdx * featureInputStride +
								inY * conv.m_InputSizeX +
								inX;
			maxValue = std::max(maxValue, input.m_Input[inputIdx]);
			assert(abs(maxValue) < 1000000.0f);
			assert(!isnan(maxValue));
			assert(!isinf(maxValue));
		}
	}
	const size_t	outIdx =	range.m_FeatureIdx * featureOutputStride +
								range.m_ConvIdxY * conv.m_OutputSizeX +
								range.m_ConvIdxX;
	input.m_Output[outIdx] = maxValue;
	assert(abs(input.m_Output[outIdx]) < 1000000.0f);
	assert(!isnan(input.m_Output[outIdx]));
	assert(!isinf(input.m_Output[outIdx]));
}

void	CLayerMaxPooling2D::Kernel_GatherSlopes(const SGatherSlopes_KernelIn &input,
												const SKernelRange &range,
												const SConvolutionParams &conv)
{
	const size_t	featureInputStride = conv.m_InputSizeX * conv.m_InputSizeY;
	const size_t	featureOutputStride = conv.m_OutputSizeX * conv.m_OutputSizeY;
	const size_t	outFeatureWeightStride = conv.m_KernelSizeX * conv.m_KernelSizeY;
	float			maxValue = -FLT_MAX;
	size_t			maxIdx = 0;

	for (size_t inY = range.m_StartConvY; inY < range.m_StopConvY; ++inY)
	{
		for (size_t inX = range.m_StartConvX; inX < range.m_StopConvX; ++inX)
		{
			size_t	inputIdx =	range.m_FeatureIdx * featureInputStride +
								inY * conv.m_InputSizeX +
								inX;
			if (input.m_Input[inputIdx] > maxValue)
			{
				maxValue = input.m_Input[inputIdx];
				maxIdx = inputIdx;
			}
		}
	}
	const size_t	outIdx =	range.m_FeatureIdx * featureOutputStride +
								range.m_ConvIdxY * conv.m_OutputSizeX +
								range.m_ConvIdxX;
	input.m_Output[maxIdx] = input.m_Slopes[outIdx];
	assert(abs(input.m_Output[maxIdx]) < 1000000.0f);
	assert(!isnan(input.m_Output[outIdx]));
	assert(!isinf(input.m_Output[outIdx]));
}

