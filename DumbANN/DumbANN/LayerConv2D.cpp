
#include "LayerConv2D.h"
#include <assert.h>

CLayerConv2D::CLayerConv2D()
:	m_InputSizeX(0)
,	m_InputSizeY(0)
,	m_FeatureCount(0)
,	m_FeatureSizeX(0)
,	m_FeatureSizeY(0)
{
}

CLayerConv2D::~CLayerConv2D()
{
}

bool	CLayerConv2D::Setup(size_t inputSizeX, size_t inputSizeY,
							size_t featureCount, size_t featureSizeX, size_t featureSizeY,
							EActivation activation,
							ERandInitializer randInit,
							EOptimization optimization)
{
	m_Activation = activation;
	m_Initializer = randInit;
	m_Optimization = optimization;

	m_InputSizeX = inputSizeX;
	m_InputSizeY = inputSizeY;
	m_FeatureCount = featureCount;
	m_FeatureSizeX = featureSizeX;
	m_FeatureSizeY = featureSizeY;
	m_InputSize = inputSizeX * inputSizeY;
#if		0
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

	memset(m_SlopesWeightAccum.Data(), 0, m_SlopesWeightAccum.StorageByteSize());
	memset(m_SlopesOutAccum.Data(), 0, m_SlopesOutAccum.Size() * sizeof(float));

	// Initialize weights to random floats:
	Initializer();
#endif
	return true;
}

void	CLayerConv2D::FeedForward(const float* input, size_t rangeMin, size_t rangeMax)
{

}

void	CLayerConv2D::BackPropagateError(const float* prevOutput, const std::vector<float>& error, size_t rangeMin, size_t rangeMax)
{

}

void	CLayerConv2D::BackPropagateError(const float* prevOutput, const CLayer* nextLayer, size_t rangeMin, size_t rangeMax)
{
	
}

void	CLayerConv2D::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{

}
