
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
	m_Output.AllocateStorage(m_FeatureCount * m_ConvParams.m_OutputSizeX * m_ConvParams.m_OutputSizeY);
	m_SlopesOut.AllocateStorage(m_FeatureCount * m_ConvParams.m_OutputSizeX * m_ConvParams.m_OutputSizeY);

	m_InputSize = inputFeatureCount * inputSizeX * inputSizeY;
	return true;
}

void	CLayerMaxPooling2D::FeedForward(const float *input, size_t rangeMin, size_t rangeMax)
{
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
	if (prevLayer == nullptr)
	{
		assert(false); // CLayerMaxPooling2D cannot be first layer
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
			assert(abs(maxValue) < 100000000.0f);
		}
	}
	const size_t	outIdx =	range.m_FeatureIdx * featureOutputStride +
								range.m_ConvIdxY * conv.m_OutputSizeX +
								range.m_ConvIdxX;
	input.m_Output[outIdx] = maxValue;
	assert(abs(input.m_Output[outIdx]) < 100000000.0f);
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
	assert(abs(input.m_Output[maxIdx]) < 100000000.0f);
}

