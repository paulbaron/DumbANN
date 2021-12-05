
#include "LayerDropout.h"
#include <assert.h>
#include <stdlib.h>

CLayerDropOut::CLayerDropOut()
:	m_Rate(0.0f)
{
}

CLayerDropOut::~CLayerDropOut()
{
}

bool	CLayerDropOut::Setup(size_t inputSize, float rate)
{
	assert(rate < 1.0f && rate > 0.0f);
	if (rate >= 1.0f || rate <= 0.0f)
		return false;

	size_t	invRate = 1.0f / rate;

	m_Rate = rate;
	m_InputSize = inputSize;
	m_OutputSize = inputSize;
	m_Output.AllocateStorage(m_InputSize);
	m_SlopesOut.AllocateStorage(m_InputSize);
	size_t	disabledIdxSize = m_InputSize / invRate;
	if (m_InputSize % invRate != 0)
		disabledIdxSize += 1;
	m_DisabledIdx.resize(disabledIdxSize);
	UpdateDisabledArray();
	return true;
}

void	CLayerDropOut::FeedForward(const float *input, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerDropOut", "CLayerDropOut::FeedForward", MP_GREEN1);
	size_t			invRate = 1.0f / m_Rate;
	const float		outScale = 1.0f / (1.0f - m_Rate);
	for (size_t i = rangeMin; i < rangeMax; ++i)
	{
		if (m_DisabledIdx[i / invRate] == i)
			m_Output.Data()[i] = 0.0f;
		else
			m_Output.Data()[i] = input[i] * outScale;
	}
}

void	CLayerDropOut::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerDropOut", "CLayerDropOut::BackPropagateError", MP_RED1);
	size_t			invRate = 1.0f / m_Rate;
	const float		*errorPtr = error.data();
	float			*slopePtr = m_SlopesOut.Data();

	for (size_t outIdx = rangeMin; outIdx < rangeMax; ++outIdx)
	{
		if (m_DisabledIdx[outIdx / invRate] == outIdx)
			slopePtr[outIdx] = 0.0f;
		else
			slopePtr[outIdx] = -errorPtr[outIdx];
	}
}

void	CLayerDropOut::BackPropagateError(const float *prevOutput, const CLayer* nextLayer, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerDropOut", "CLayerDropOut::BackPropagateError", MP_RED1);
	size_t			invRate = 1.0f / m_Rate;
	float			*slopePtr = m_SlopesOut.Data();

	for (size_t outIdx = rangeMin; outIdx < rangeMax; ++outIdx)
	{
		if (m_DisabledIdx[outIdx / invRate] == outIdx)
			slopePtr[outIdx] = 0.0f;
	}
}

void	CLayerDropOut::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerDropOut", "CLayerDropOut::UpdateDisabledArray", MP_BLUE1);
	UpdateDisabledArray();
}

void	CLayerDropOut::GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const
{
	MICROPROFILE_SCOPEI("CLayerDropOut", "CLayerDropOut::GatherSlopes", MP_PALEVIOLETRED1);
	size_t			invRate = 1.0f / m_Rate;
	for (size_t i = rangeMin; i < rangeMax; ++i)
	{
		if (m_DisabledIdx[i / invRate] == i)
			dst[i] = 0.0f;
		else
			dst[i] = m_SlopesOut.Data()[i];
	}
}

void	CLayerDropOut::PrintInfo() const
{
	printf("\tLayer DropOut:\n");
	printf("\t\tInput: %zu\n", m_InputSize);
	printf(	"\t\tRate: %f\n", m_Rate);
}

void	CLayerDropOut::Serialize(std::vector<uint8_t>& data) const
{
	SerializeLayerType(data, ELayerType::LayerDropout);
	SerializeInOutSize(data);
	size_t		prevSize = data.size();
	data.resize(prevSize + sizeof(float));
	float		*dataPtr = (float*)(data.data() + prevSize);
	dataPtr[0] = m_Rate;
}

bool	CLayerDropOut::UnSerialize(const std::vector<uint8_t> &data, size_t &curIdx)
{
	if (!UnSerializeInOutSize(data, curIdx))
		return false;
	if (curIdx + sizeof(float) > data.size())
		return false;
	float		*dataPtr = (float*)(data.data() + curIdx);
	m_Rate = dataPtr[0];
	curIdx += sizeof(float);
	if (!Setup(m_InputSize, m_Rate))
		return false;
	return true;
}

size_t	CLayerDropOut::GetThreadingHint() const
{
	return 1;
}

size_t	CLayerDropOut::GetDomainSize() const
{
	return GetOutputSize();
}

void	CLayerDropOut::UpdateDisabledArray()
{
	size_t		invRate = 1.0f / m_Rate;
	const float	seed = (float)rand() / (float)RAND_MAX;
	bool		neuronDisabled = false;

	for (size_t i = 0; i < m_InputSize; ++i)
	{
		float	fastRand = sinf((float)i + seed) * 58492.47388f;
		fastRand = fastRand - floorf(fastRand);

		if ((i + 1) % invRate == 0)
		{
			if (!neuronDisabled)
				m_DisabledIdx[i / invRate] = floor(RemapValue(fastRand, 0.0f, 1.0f, (i + 1) - invRate, i + 1));
			neuronDisabled = false;
		}
		else if (fastRand < m_Rate && !neuronDisabled)
		{
			m_DisabledIdx[i / invRate] = i;
			neuronDisabled = true;
		}
	}
}
