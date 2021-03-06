
#include "LayerDense.h"
#include <assert.h>

CLayerDense::CLayerDense()
{
}

CLayerDense::~CLayerDense()
{
}

bool	CLayerDense::Setup(size_t inputSize, size_t outputSize)
{
	m_InputSize = inputSize;
	m_OutputSize = outputSize;
	m_Weights.AllocMatrix(outputSize, inputSize);
	m_SlopesWeightAccum.AllocMatrix(outputSize, inputSize);
	m_SlopesOutAccum.AllocateStorage(outputSize);
	m_Bias.AllocateStorage(outputSize);

	// When using inertia we need those storages:
	m_DeltaWeightVelocity.AllocMatrix(outputSize, inputSize);
	m_DeltaBiasVelocity.AllocateStorage(outputSize);
	m_NetInput.AllocateStorage(outputSize);
	m_Output.AllocateStorage(outputSize);
	m_SlopesOut.AllocateStorage(outputSize);

	m_AdagradWeightAccum.AllocMatrix(outputSize, inputSize);
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

	memset(m_SlopesWeightAccum.Data(), 0, m_SlopesWeightAccum.StorageByteSize());
	memset(m_SlopesOutAccum.Data(), 0, m_SlopesOutAccum.Size() * sizeof(float));
	
	// Initialize weights to random floats:
	Initializer();
	return true;
}

void	CLayerDense::FeedForward(const float *input, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerDense", "CLayerDense::FeedForward", MP_GREEN1);
	assert(rangeMin >= 0 && rangeMin < m_Output.Size() && rangeMin < rangeMax);
	assert(rangeMax >= 0 && rangeMax <= m_Output.Size());
	const size_t			outputRange = rangeMax - rangeMin;
	float					*outputPtr = m_Output.Data() + rangeMin;
	float					*netInputPtr = m_NetInput.Data() + rangeMin;
	const float				*biasesPtr = m_Bias.Data() + rangeMin;
	const float				*weightsPtr = m_Weights.View().GetRow(rangeMin);
	SConstNeuronMatrixView	weightMat(weightsPtr, outputRange, m_InputSize, m_Weights.View().m_RowByteStride);

	// MatrixMAdd computes net input:
	CNeuronMatrix::ComputeNetInput(netInputPtr, input, weightMat, biasesPtr);
	Activation(outputPtr, netInputPtr, outputRange);
}

void	CLayerDense::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerDense", "CLayerDense::BackPropagateError", MP_RED1);
	assert(rangeMin >= 0 && rangeMin < m_Output.Size() && rangeMin < rangeMax);
	assert(rangeMax >= 0 && rangeMax <= m_Output.Size());
	const size_t	outputRange = rangeMax - rangeMin;
	float			*slopePtr = m_SlopesOut.Data();
	float			*netInputPtr = m_NetInput.Data();
	const float		*errorPtr = error.data();

	// Outter layer of the neural network:
	// Cost derivative:
	for (size_t outIdx = rangeMin; outIdx < rangeMax; ++outIdx)
		slopePtr[outIdx] = -errorPtr[outIdx];
	// Activation derivative:
	ActivationDerivative(slopePtr + rangeMin, netInputPtr + rangeMin, outputRange);
	if (m_Learn)
	{
		// We compute the delta for the weights and bias (for the bias its just the output slope):
		float		*slopeAccumPtr = m_SlopesOutAccum.Data();
		for (size_t outIdx = rangeMin; outIdx < rangeMax; ++outIdx)
		{
			float		*slopeWeightAccumPtr = m_SlopesWeightAccum.View().GetRow(outIdx);
			slopeAccumPtr[outIdx] += slopePtr[outIdx];
			for (size_t inIdx = 0; inIdx < m_InputSize; ++inIdx)
				slopeWeightAccumPtr[inIdx] += slopePtr[outIdx] * prevOutput[inIdx];
		}
	}
}

void	CLayerDense::BackPropagateError(const float *prevOutput, const CLayer *nextLayer, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerDense", "CLayerDense::BackPropagateError", MP_RED1);
	assert(rangeMin >= 0 && rangeMin < m_Output.Size() && rangeMin < rangeMax);
	assert(rangeMax >= 0 && rangeMax <= m_Output.Size());
	const size_t			outputRange = rangeMax - rangeMin;
	float					*slopePtr = m_SlopesOut.Data();
	float					*netInputPtr = m_NetInput.Data();

	// Inner layer of the neural network:
	ActivationDerivative(slopePtr + rangeMin, netInputPtr + rangeMin, outputRange);
	if (m_Learn)
	{
		// We compute the delta for the weights and bias (for the bias its just the output slope):
		float	*slopeAccumPtr = m_SlopesOutAccum.Data();
		for (size_t outIdx = rangeMin; outIdx < rangeMax; ++outIdx)
		{
			float* slopeWeightAccumPtr = m_SlopesWeightAccum.View().GetRow(outIdx);
			slopeAccumPtr[outIdx] += slopePtr[outIdx];
			for (size_t inIdx = 0; inIdx < m_InputSize; ++inIdx)
				slopeWeightAccumPtr[inIdx] += slopePtr[outIdx] * prevOutput[inIdx];
		}
	}
}

void	CLayerDense::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
	MICROPROFILE_SCOPEI("CLayerDense", "CLayerDense::UpdateWeightsAndBias", MP_BLUE1);
	assert(rangeMin >= 0 && rangeMin < m_Output.Size() && rangeMin < rangeMax);
	assert(rangeMax >= 0 && rangeMax <= m_Output.Size());
	const size_t		outputRange = rangeMax - rangeMin;
	float				*slopeAccumPtr = m_SlopesOutAccum.Data();
	float				*biasesPtr = m_Bias.Data();

	OptimizeWeight(rangeMin, rangeMax, trainingSteps);
	OptimizeBias(biasesPtr, slopeAccumPtr, rangeMin, rangeMax, trainingSteps);
	memset(m_SlopesWeightAccum.View().GetRow(rangeMin), 0, outputRange * m_SlopesWeightAccum.View().m_RowByteStride);
	memset(m_SlopesOutAccum.Data() + rangeMin, 0, outputRange * sizeof(float));
}

void	CLayerDense::GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const
{
	MICROPROFILE_SCOPEI("CLayerDense", "CLayerDense::GatherSlopes", MP_PALEVIOLETRED1);
	(void)prevLayer;
	SConstNeuronMatrixView	weightMat(m_Weights.View());
	weightMat.m_Data += rangeMin;
	weightMat.m_Columns = rangeMax - rangeMin;
	CNeuronMatrix::ComputeError(dst + rangeMin, m_SlopesOut.Data(), weightMat);
}

void	CLayerDense::PrintInfo() const
{
	printf("\tLayer Dense:\n");
	printf("\t\tInput: %zu\n", m_InputSize);
	printf("\t\tOutput: %zu\n", m_Output.Size());
	PrintBasicInfo();
}

void	CLayerDense::Serialize(std::vector<uint8_t> &data) const
{
	SerializeLayerType(data, ELayerType::LayerDense);
	SerializeInOutSize(data);
	SerializeBasicInfo(data);
	SerializeWeightsAndBias(data);
}

bool	CLayerDense::UnSerialize(const std::vector<uint8_t> &data, size_t &curIdx)
{
	if (!UnSerializeInOutSize(data, curIdx))
		return false;
	if (!UnSerializeBasicInfo(data, curIdx))
		return false;
	if (!Setup(m_InputSize, m_OutputSize))
		return false;
	if (!UnSerializeWeightsAndBias(data, curIdx))
		return false;
	return true;
}

size_t	CLayerDense::GetThreadingHint() const
{
	return m_Weights.View().m_Columns * m_Weights.View().m_Rows;
}

size_t	CLayerDense::GetDomainSize() const
{
	return GetOutputSize();
}
