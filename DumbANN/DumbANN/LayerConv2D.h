#pragma once

#include "LayerBase.h"
#include "NeuronKernel.h"

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
	virtual void	GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const override;
	virtual void	PrintInfo() const override;

	virtual size_t	GetThreadingHint() const override;
	virtual size_t	GetDomainSize() const override;

	size_t			GetFeatureCount() const { return m_KernelCount; }
	size_t			GetFeatureSizeX() const { return m_ConvParams.m_KernelSizeX; }
	size_t			GetFeatureSizeY() const { return m_ConvParams.m_KernelSizeY; }
	size_t			GetOutputSizeX() const { return m_ConvParams.m_OutputSizeX; }
	size_t			GetOutputSizeY() const { return m_ConvParams.m_OutputSizeY; }

private:
	struct	SComputeNetInput_KernelIn
	{
		// Input data:
		const float				*m_Input;
		// Neuron data:
		SConstNeuronMatrixView	m_Weights;
		const float				*m_Bias;
		float					*m_NetInput;

		size_t					m_InFeatureCount;
		size_t					m_OutFeatureCount;
	};

	struct	SAccumWeightsAndBiasDerivative_KernelIn
	{
		// Input data:
		const float				*m_Input;
		// Neuron data:
		const float				*m_Slopes;
		SNeuronMatrixView		m_AccumWeights;
		float					*m_AccumBias;

		size_t					m_InFeatureCount;
		size_t					m_OutFeatureCount;
	};

	struct	SGatherSlopes_KernelIn
	{
		// Output data:
		float					*m_Output;
		// Neuron data:
		const float				*m_Slopes;
		SNeuronMatrixView		m_Weights;

		size_t					m_InFeatureCount;
		size_t					m_OutFeatureCount;
	};

	__forceinline static void		Kernel_AccumWeightsAndBiasDerivative(	const SAccumWeightsAndBiasDerivative_KernelIn &input,
																			const SKernelRange &range,
																			const SConvolutionParams &conv);
	__forceinline static void		Kernel_ComputeNetInput(	const SComputeNetInput_KernelIn &input,
															const SKernelRange &range,
															const SConvolutionParams &conv);
	__forceinline static void		Kernel_GatherSlopes(const SGatherSlopes_KernelIn &input,
														const SKernelRange &range,
														const SConvolutionParams &conv);

	SConvolutionParams	m_ConvParams;
	size_t				m_KernelCount;
	size_t				m_InputImageCount;
};