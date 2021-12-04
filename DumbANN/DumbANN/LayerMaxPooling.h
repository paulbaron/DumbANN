#pragma once

#include "LayerBase.h"
#include "NeuronKernel.h"

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
	virtual void	GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const override;
	virtual void	PrintInfo() const override;

	virtual size_t	GetThreadingHint() const override;
	virtual size_t	GetDomainSize() const override;

	size_t			GetFeatureCount() const { return m_FeatureCount; }
	size_t			GetFeatureSizeX() const { return m_ConvParams.m_KernelSizeX; }
	size_t			GetFeatureSizeY() const { return m_ConvParams.m_KernelSizeY; }
	size_t			GetOutputSizeX() const { return m_ConvParams.m_OutputSizeX; }
	size_t			GetOutputSizeY() const { return m_ConvParams.m_OutputSizeY; }

private:
	struct	SComputeOutput_KernelIn
	{
		const float				*m_Input;
		float					*m_Output;

		size_t					m_FeatureCount;
	};

	struct	SGatherSlopes_KernelIn
	{
		const float				*m_Input;
		float					*m_Output;

		const float				*m_Slopes;

		size_t					m_FeatureCount;
	};

	__forceinline static void		Kernel_ComputeOutput(	const SComputeOutput_KernelIn &input,
															const SKernelRange &range,
															const SConvolutionParams &conv);
	__forceinline static void		Kernel_GatherSlopes(const SGatherSlopes_KernelIn &input,
														const SKernelRange &range,
														const SConvolutionParams &conv);

	SConvolutionParams	m_ConvParams;
	size_t				m_FeatureCount;
};