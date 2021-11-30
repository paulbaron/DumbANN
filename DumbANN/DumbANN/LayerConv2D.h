#pragma once

#include "LayerBase.h"

class	CLayerConv2D : public CLayer
{
public:
	CLayerConv2D();
	~CLayerConv2D();

	bool	Setup(	size_t inputFeatureCount, size_t inputSizeX, size_t inputSizeY,
					size_t featureCount, size_t featureSizeX, size_t featureSizeY,
					EActivation activation = EActivation::Sigmoid,
					ERandInitializer randInit = ERandInitializer::RandXavier,
					EOptimization optimization = EOptimization::SGD);

	virtual void	FeedForward(const float *input, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float* prevOutput, const CLayer *nextLayer, size_t rangeMin, size_t rangeMax) override;
	virtual void	UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax) override;
	virtual void	GatherSlopes(float *dst, size_t rangeMin, size_t rangeMax) const override;

	virtual size_t	GetThreadingHint() const override;
	virtual size_t	GetDomainSize() const override;

private:
	size_t	m_InputFeatureCount;
	size_t	m_InputSizeX;
	size_t	m_InputSizeY;
	size_t	m_FeatureCount;
	size_t	m_FeatureSizeX;
	size_t	m_FeatureSizeY;
};