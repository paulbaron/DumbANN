
#include "LayerSoftmax.h"
#include <assert.h>
#include <stdlib.h>

CLayerSoftMax::CLayerSoftMax()
:	m_CurrentSum(0.0f)
{
}

CLayerSoftMax::~CLayerSoftMax()
{
}

bool	CLayerSoftMax::Setup(size_t inputSize)
{
	m_InputSize = inputSize;
	m_Output.AllocateStorage(m_InputSize);
	m_SlopesOut.AllocateStorage(m_InputSize);
	return true;
}

void	CLayerSoftMax::FeedForward(const float *input, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerSoftMax", "CLayerSoftMax::FeedForward", MP_GREEN1);
	assert(rangeMin == 0);
	assert(rangeMax == m_InputSize);
	m_CurrentSum = 0.0f;
	for (size_t i = 0; i < m_InputSize; i++)
		m_CurrentSum += exp(input[i]);
	for (size_t i = 0; i < m_InputSize; i++)
		m_Output.Data()[i] = exp(input[i]) / m_CurrentSum;
}

void	CLayerSoftMax::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerSoftMax", "CLayerSoftMax::BackPropagateError", MP_RED1);
	float			*slopePtr = m_SlopesOut.Data();
	const float		*errorPtr = error.data();

	for (size_t outIdx = rangeMin; outIdx < rangeMax; ++outIdx)
		slopePtr[outIdx] = -errorPtr[outIdx];
	for (size_t i = rangeMin; i < rangeMax; i++)
	{
		float	softmaxValue = exp(prevOutput[i]) / m_CurrentSum;
		slopePtr[i] *= softmaxValue * (1.0f - softmaxValue);
	}
}

void	CLayerSoftMax::BackPropagateError(const float *prevOutput, const CLayer* nextLayer, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerSoftMax", "CLayerSoftMax::BackPropagateError", MP_RED1);
	float			*slopePtr = m_SlopesOut.Data();
	for (size_t i = rangeMin; i < rangeMax; i++)
	{
		float	softmaxValue = exp(prevOutput[i]) / m_CurrentSum;
		slopePtr[i] *= softmaxValue * (1.0f - softmaxValue);
		assert(abs(slopePtr[i]) < 1000000.0f);
	}
}

void	CLayerSoftMax::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
}

void	CLayerSoftMax::GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const
{
	MICROPROFILE_SCOPEI("CLayerSoftMax", "CLayerSoftMax::GatherSlopes", MP_PALEVIOLETRED1);
	float			*slopePtr = m_SlopesOut.Data();
	for (size_t i = rangeMin; i < rangeMax; ++i)
	{
		dst[i] = slopePtr[i];
		assert(abs(dst[i]) < 1000000.0f);
	}
}

void	CLayerSoftMax::PrintInfo() const
{
	printf("\tLayer Softmax:\n");
	printf("\t\tInput: %zu\n", m_InputSize);
}

size_t	CLayerSoftMax::GetThreadingHint() const
{
	return 1;
}

size_t	CLayerSoftMax::GetDomainSize() const
{
	return GetOutputSize();
}
