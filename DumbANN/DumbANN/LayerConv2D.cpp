
#include "LayerConv2D.h"
#include <assert.h>

CLayerConv2D::CLayerConv2D()
:	m_InputFeatureCount(0)
,	m_InputSizeX(0)
,	m_InputSizeY(0)
,	m_FeatureCount(0)
,	m_FeatureSizeX(0)
,	m_FeatureSizeY(0)
{
}

CLayerConv2D::~CLayerConv2D()
{
}

bool	CLayerConv2D::Setup(size_t inputFeatureCount, size_t inputSizeX, size_t inputSizeY,
							size_t featureCount, size_t featureSizeX, size_t featureSizeY,
							EActivation activation,
							ERandInitializer randInit,
							EOptimization optimization)
{
	m_Activation = activation;
	m_Initializer = randInit;
	m_Optimization = optimization;

	m_InputFeatureCount = inputFeatureCount;
	m_InputSizeX = inputSizeX;
	m_InputSizeY = inputSizeY;
	m_FeatureCount = featureCount;
	m_FeatureSizeX = featureSizeX;
	m_FeatureSizeY = featureSizeY;

	const size_t	featureOutputSizeX = (inputSizeX + 1) - featureSizeX;
	const size_t	featureOutputSizeY = (inputSizeY + 1) - featureSizeY;
	const size_t	outputSize = featureCount * featureOutputSizeX * featureOutputSizeY;
	const size_t	weightsSizeX = m_InputFeatureCount * featureSizeX * featureSizeY;
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
	const size_t	featureOutputSizeX = (m_InputSizeX + 1) - m_FeatureSizeX;
	const size_t	featureOutputSizeY = (m_InputSizeY + 1) - m_FeatureSizeY;
	const size_t	featureInputStride = m_InputSizeX * m_InputSizeY;
	const size_t	outFeatureWeightStride = m_FeatureSizeX * m_FeatureSizeY;

	// For each feature:
	for (size_t outFeatureIdx = rangeMin; outFeatureIdx < rangeMax; ++outFeatureIdx)
	{
		const float		*weightsPtr = m_Weights.View().GetRow(outFeatureIdx);
		// Convolution:
		for (size_t convY = 0; convY < featureOutputSizeY; ++convY)
		{
			for (size_t convX = 0; convX < featureOutputSizeX; ++convX)
			{
				float		accum = 0.0f;
				// For each input feature:
				for (size_t inFeatureIdx = 0; inFeatureIdx < m_InputFeatureCount; ++inFeatureIdx)
				{
					for (size_t inY = 0; inY < m_FeatureSizeY; ++inY)
					{
						for (size_t inX = 0; inX < m_FeatureSizeX; ++inX)
						{
							size_t	inputIdx =	inFeatureIdx * featureInputStride +
												(inY + convY) * m_InputSizeX +
												(inX + convX);
							size_t	weightIdx = inFeatureIdx * outFeatureWeightStride + 
												inY * m_FeatureSizeX + inX;
							accum += input[inputIdx] * weightsPtr[weightIdx];
						}
					}
				}
				const size_t	outIdx =	outFeatureIdx * featureOutputSizeX * featureOutputSizeY +
											convY * featureOutputSizeX +
											convX;
				accum += m_Bias.Data()[outIdx];
				m_NetInput.Data()[outIdx] = accum;
			}
		}
	}
	const size_t	featureStide = featureOutputSizeX * featureOutputSizeY;
	const size_t	outputRange = (rangeMax - rangeMin) * featureStide;
	Activation(m_Output.Data() + featureStide * rangeMin, m_NetInput.Data() + featureStide * rangeMin, outputRange);
}

void	CLayerConv2D::BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax)
{
	// Not supported as output yet...
	assert(false);
}

void	CLayerConv2D::BackPropagateError(const float *prevOutput, const CLayer* nextLayer, size_t rangeMin, size_t rangeMax)
{
	const size_t	featureOutputSizeX = (m_InputSizeX + 1) - m_FeatureSizeX;
	const size_t	featureOutputSizeY = (m_InputSizeY + 1) - m_FeatureSizeY;
	const size_t	featureInputStride = m_InputSizeX * m_InputSizeY;
	const size_t	outFeatureWeightStride = m_FeatureSizeX * m_FeatureSizeY;
	const size_t	featureStide = featureOutputSizeX * featureOutputSizeY;
	const size_t	outputRange = (rangeMax - rangeMin) * featureStide;
	float			*slopePtr = m_SlopesOut.Data();
	float			*netInputPtr = m_NetInput.Data();

	// Inner layer of the neural network:
	nextLayer->GatherSlopes(m_SlopesOut.Data(), rangeMin, rangeMax);
	ActivationDerivative(m_SlopesOut.Data() + featureStide * rangeMin, netInputPtr + featureStide * rangeMin, outputRange);
	// We compute the delta for the weights and bias (for the bias its just the output slope):
	// For each feature:
	for (size_t outFeatureIdx = rangeMin; outFeatureIdx < rangeMax; ++outFeatureIdx)
	{
		float		*slopeWeightAccumPtr = m_SlopesWeightAccum.View().GetRow(outFeatureIdx);
		// Convolution:
		for (size_t convY = 0; convY < featureOutputSizeY; ++convY)
		{
			for (size_t convX = 0; convX < featureOutputSizeX; ++convX)
			{
				const size_t	outIdx =	outFeatureIdx * featureOutputSizeX * featureOutputSizeY +
											convY * featureOutputSizeX +
											convX;
				// For each input feature:
				for (size_t inFeatureIdx = 0; inFeatureIdx < m_InputFeatureCount; ++inFeatureIdx)
				{
					for (size_t inY = 0; inY < m_FeatureSizeY; ++inY)
					{
						for (size_t inX = 0; inX < m_FeatureSizeX; ++inX)
						{
							size_t	inputIdx =	inFeatureIdx * featureInputStride +
												(inY + convY) * m_InputSizeX +
												(inX + convX);
							size_t	weightIdx = inFeatureIdx * outFeatureWeightStride + 
												inY * m_FeatureSizeX + inX;
							slopeWeightAccumPtr[weightIdx] += prevOutput[inputIdx] * slopePtr[outIdx];
						}
					}
				}
			}
		}
	}
}

void	CLayerConv2D::UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax)
{
	const size_t	outputRange = rangeMax - rangeMin;
	float			*slopeAccumPtr = m_SlopesOutAccum.Data();
	float			*biasesPtr = m_Bias.Data();
	const size_t	featureOutputSizeX = (m_InputSizeX + 1) - m_FeatureSizeX;
	const size_t	featureOutputSizeY = (m_InputSizeY + 1) - m_FeatureSizeY;
	const size_t	featureOutputStride = featureOutputSizeX * featureOutputSizeY;

	OptimizeWeight(rangeMin, rangeMax, trainingSteps);
	OptimizeBias(biasesPtr, slopeAccumPtr, rangeMin * featureOutputStride, rangeMax * featureOutputStride, trainingSteps);

	memset(m_SlopesWeightAccum.View().GetRow(rangeMin), 0, outputRange * m_SlopesWeightAccum.View().m_RowByteStride);
	memset(m_SlopesOutAccum.Data() + rangeMin * featureOutputStride, 0, outputRange * featureOutputStride * sizeof(float));
}

void	CLayerConv2D::GatherSlopes(float *dst, size_t rangeMin, size_t rangeMax) const
{
	const size_t	dstRange = rangeMax - rangeMin;
	const size_t	featureOutputSizeX = (m_InputSizeX + 1) - m_FeatureSizeX;
	const size_t	featureOutputSizeY = (m_InputSizeY + 1) - m_FeatureSizeY;
	const size_t	featureInputStride = m_InputSizeX * m_InputSizeY;
	const size_t	featureWeightStride = m_FeatureSizeX * m_FeatureSizeY;

	// Init to zero, will add all derivative for each output neuron:
	memset(dst + rangeMin * featureInputStride, 0, dstRange * featureInputStride);
	// For each input feature:
	for (size_t inFeatureIdx = rangeMin; inFeatureIdx < rangeMax; ++inFeatureIdx)
	{
		// For each output feature:
		for (size_t outFeatureIdx = 0; outFeatureIdx < m_FeatureCount; ++outFeatureIdx)
		{
			const float		*weightsPtr = m_Weights.View().GetRow(outFeatureIdx);
			// Convolution:
			for (size_t convY = 0; convY < featureOutputSizeY; ++convY)
			{
				for (size_t convX = 0; convX < featureOutputSizeX; ++convX)
				{
					const size_t	outIdx =	outFeatureIdx * featureOutputSizeX * featureOutputSizeY +
												convY * featureOutputSizeX +
												convX;
					float		slope = m_SlopesOut.Data()[outIdx];
					for (size_t inY = 0; inY < m_FeatureSizeY; ++inY)
					{
						for (size_t inX = 0; inX < m_FeatureSizeX; ++inX)
						{
							size_t	dstIdx =	inFeatureIdx * featureInputStride +
												(inY + convY) * m_InputSizeX +
												(inX + convX);
							size_t	weightIdx = inFeatureIdx * featureWeightStride +
												inY * m_FeatureSizeX + inX;
							dst[dstIdx] += slope * weightsPtr[weightIdx];
						}
					}
				}
			}
		}
	}
}

size_t	CLayerConv2D::GetThreadingHint() const
{
	return m_Weights.View().m_Columns * m_Weights.View().m_Rows;
}

size_t	CLayerConv2D::GetDomainSize() const
{
	return m_FeatureCount;
}
