
#include "LayerDense.h"
#include <assert.h>

CLayerDense::CLayerDense()
{
}

CLayerDense::~CLayerDense()
{
}

bool	CLayerDense::Setup(size_t inputSize, size_t outputSize,
							EActivation activation,
							ERandInitializer randInit,
							EOptimization optimization)
{
	m_Activation = activation;
	m_Initializer = randInit;
	m_Optimization = optimization;

	m_InputSize = inputSize;
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
	assert(rangeMin >= 0 && rangeMin < m_Output.Size() && rangeMin < rangeMax);
	assert(rangeMax >= 0 && rangeMax <= m_Output.Size());
	const size_t			outputRange = rangeMax - rangeMin;
	float					*outputPtr = m_Output.Data() + rangeMin;
	float					*netInputPtr = m_NetInput.Data() + rangeMin;
	const float				*biasesPtr = m_Bias.Data() + rangeMin;
	const float				*weightsPtr = m_Weights.View().GetRow(rangeMin);
	SConstNeuronMatrixView	weightMat(weightsPtr, outputRange, m_InputSize, m_Weights.View().m_RowStride);

	// MatrixMAdd computes net input:
	CNeuronMatrix::ComputeNetInput(netInputPtr, input, weightMat, biasesPtr);
	Activation(outputPtr, netInputPtr, outputRange);
}

void	CLayerDense::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
	assert(rangeMin >= 0 && rangeMin < m_Output.Size() && rangeMin < rangeMax);
	assert(rangeMax >= 0 && rangeMax <= m_Output.Size());
	const size_t	outputRange = rangeMax - rangeMin;
	float			*slopePtr = m_SlopesOut.Data();
	float			*netInputPtr = m_NetInput.Data();
	const float		*errorPtr = error.data();

	// Outter layer of the neural network:
	// Activation derivative:
	ActivationDerivative(slopePtr + rangeMin, netInputPtr + rangeMin, outputRange);
	// Cost derivative:
	for (size_t outIdx = rangeMin; outIdx < rangeMax; ++outIdx)
		slopePtr[outIdx] *= -errorPtr[outIdx];
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

void	CLayerDense::BackPropagateError(const float *prevOutput, const CLayer *nextLayer, size_t rangeMin, size_t rangeMax)
{
	assert(rangeMin >= 0 && rangeMin < m_Output.Size() && rangeMin < rangeMax);
	assert(rangeMax >= 0 && rangeMax <= m_Output.Size());
	const size_t			outputRange = rangeMax - rangeMin;
	float					*slopePtr = m_SlopesOut.Data();
	float					*netInputPtr = m_NetInput.Data();
	const size_t			nextNeuronCount = nextLayer->GetWeights().View().m_Rows;
	const float				*nextSlopePtr = nextLayer->GetSlopesOut().Data();
	const float				*nextWeightPtr = nextLayer->GetWeights().View().m_Data + rangeMin;
	SConstNeuronMatrixView	weightMat(nextWeightPtr, nextNeuronCount, outputRange, nextLayer->GetWeights().View().m_RowStride);

	// Inner layer of the neural network:
	ActivationDerivative(slopePtr + rangeMin, netInputPtr + rangeMin, outputRange);
	CNeuronMatrix::ComputeError(slopePtr + rangeMin, nextSlopePtr, weightMat);
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

void	CLayerDense::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
	assert(rangeMin >= 0 && rangeMin < m_Output.Size() && rangeMin < rangeMax);
	assert(rangeMax >= 0 && rangeMax <= m_Output.Size());
	const size_t		outputRange = rangeMax - rangeMin;
	float				*slopeAccumPtr = m_SlopesOutAccum.Data();
	float				*biasesPtr = m_Bias.Data();

	OptimizeWeight(rangeMin, rangeMax, trainingSteps);
	OptimizeBias(biasesPtr, slopeAccumPtr, rangeMin, rangeMax, trainingSteps);
	memset(m_SlopesWeightAccum.View().GetRow(rangeMin), 0, outputRange * m_SlopesWeightAccum.View().m_RowStride);
	memset(m_SlopesOutAccum.Data() + rangeMin, 0, outputRange * sizeof(float));
}
