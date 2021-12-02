#pragma once

#include "LayerBase.h"

class	CLayerMaxPooling2D : public CLayer
{
public:
	CLayerMaxPooling2D();
	~CLayerMaxPooling2D();

	bool	Setup(	size_t inputFeatureCount, size_t inputSizeX, size_t inputSizeY,
					size_t poolSizeX, size_t poolSizeY,
					size_t padding, size_t stride);

	virtual void	FeedForward(const float *input, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float* prevOutput, const CLayer *nextLayer, size_t rangeMin, size_t rangeMax) override;
	virtual void	UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax) override;
	virtual void	GatherSlopes(float *dst, size_t rangeMin, size_t rangeMax) const override;

	virtual size_t	GetThreadingHint() const override;
	virtual size_t	GetDomainSize() const override;

private:
	void			AccumWeightsAndBiasDerivative(const float *prevOutput, size_t rangeMin, size_t rangeMax);

	SConvolutionParams	m_ConvParams;
};