#pragma once

#include "LayerBase.h"
#include "NeuronKernel.h"

class	CLayerDropOut : public CLayer
{
public:
	CLayerDropOut();
	~CLayerDropOut();

	bool	Setup(size_t inputSize, float rate);

	virtual void	FeedForward(const float *input, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float* prevOutput, const CLayer *nextLayer, size_t rangeMin, size_t rangeMax) override;
	virtual void	UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax) override;
	virtual void	GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const override;
	virtual void	PrintInfo() const override;

	virtual size_t	GetThreadingHint() const override;
	virtual size_t	GetDomainSize() const override;

private:
	float				m_Rate;
	std::vector<size_t>	m_DisabledIdx;
};