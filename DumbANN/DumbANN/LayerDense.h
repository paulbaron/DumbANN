#pragma once

#include "LayerBase.h"

class	CLayerDense : public CLayer
{
public:
	CLayerDense();
	~CLayerDense();

	bool	Setup(size_t inputSize, size_t outputSize);

	virtual void	FeedForward(const float *input, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float *prevOutput, const CLayer *nextLayer, size_t rangeMin, size_t rangeMax) override;
	virtual void	UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax) override;
	virtual void	GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const override;
	virtual void	PrintInfo() const override;
	virtual void	Serialize(std::vector<uint8_t> &data) const override;
	virtual bool	UnSerialize(const std::vector<uint8_t> &data, size_t &curIdx) override;

	virtual size_t	GetThreadingHint() const override;
	virtual size_t	GetDomainSize() const override;
};
