#pragma once

#include "LayerBase.h"

class	CLayerConv2D : public CLayer
{
public:
	CLayerConv2D();
	~CLayerConv2D();

	bool	Setup(	size_t inputFeatureCount, size_t inputSizeX, size_t inputSizeY,
					size_t featureCount, size_t featureSizeX, size_t featureSizeY,
					size_t padding, size_t stride);

	virtual void	FeedForward(const float *input, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax) override;
	virtual void	BackPropagateError(const float* prevOutput, const CLayer *nextLayer, size_t rangeMin, size_t rangeMax) override;
	virtual void	UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax) override;
	virtual void	GatherSlopes(float *dst, size_t rangeMin, size_t rangeMax) const override;

	virtual size_t	GetThreadingHint() const override;
	virtual size_t	GetDomainSize() const override;

	size_t			GetFeatureCount() const { return m_ConvParams.m_KernelCount; }
	size_t			GetFeatureSizeX() const { return m_ConvParams.m_KernelSizeX; }
	size_t			GetFeatureSizeY() const { return m_ConvParams.m_KernelSizeY; }
	size_t			GetOutputSizeX() const { return m_ConvParams.GetConvOutputSizeX(); }
	size_t			GetOutputSizeY() const { return m_ConvParams.GetConvOutputSizeY(); }

private:
	void			Kernel_AccumWeightsAndBiasDerivative(	const float *input,
															const SConvolutionParams &conv,
															size_t featureIdx,
															int startInX, int stopInX,
															int startInY, int stopInY,
															int convX, int convY,
															int paddedConvX, int paddedConvY);
	void			Kernel_ComputeNetInput(	const float *input,
											const SConvolutionParams &conv,
											size_t featureIdx,
											int startInX, int stopInX,
											int startInY, int stopInY,
											int convX, int convY,
											int paddedConvX, int paddedConvY);

	void			Kernel_GatherSlopes(	float *output,
											const SConvolutionParams &conv,
											size_t inFeatureIdx, size_t outFeatureIdx,
											int startInX, int stopInX,
											int startInY, int stopInY,
											int convX, int convY,
											int paddedConvX, int paddedConvY) const;

	SConvolutionParams	m_ConvParams;
};