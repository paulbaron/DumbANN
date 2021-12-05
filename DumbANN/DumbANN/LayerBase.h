#pragma once

#include "DumbANNConfig.h"
#include "NeuronStorages.h"
#include <vector>

float	RemapValue(float value, float oldMin, float oldMax, float newMin, float newMax);

enum class	EActivation
{
	Relu,
	LeakyRelu,
	Elu,
	Gelu,
	Sigmoid,
	Tanh,
	Linear
};

enum class	EOptimization
{
	SGD,
	Adagrad
};
enum class	ERegularizer
{
	None,
	L1,
	L2
};

enum class	ERandInitializer
{
	RandUniform_0_1,
	RandUniform_MINUS1_1,
	RandXavier,
	RandXavierNormalized,
	RandHe
};

const char	*kActivationNames[];
const char	*kOptimizationNames[];
const char	*kRegularizationNames[];
const char	*kInitializerNames[];

enum class	ELayerType
{
	LayerDense,
	LayerConv2D,
	LayerMaxPooling,
	LayerDropout,
	LayerSofmax
};

class	CLayer
{
public:
	static CLayer	*CreateLayer(ELayerType type);

	CLayer();
	~CLayer();

	size_t	GetInputSize() const { return m_InputSize; }
	size_t	GetOutputSize() const { return m_Output.Size(); }
	const CNeuronVector			&GetOutput() const { return m_Output; }
	const CNeuronVector			&GetNetInput() const { return m_NetInput; }
	const CNeuronMatrix			&GetWeights() const { return m_Weights; }
	const CNeuronVector			&GetSlopesOut() const { return m_SlopesOut; }

	virtual void	FeedForward(const float *input, size_t rangeMin, size_t rangeMax) = 0;
	virtual void	BackPropagateError(const float *prevOutput, const std::vector<float> &error, size_t rangeMin, size_t rangeMax) = 0;
	virtual void	BackPropagateError(const float* prevOutput, const CLayer *nextLayer, size_t rangeMin, size_t rangeMax) = 0;
	virtual void	UpdateWeightsAndBias(size_t trainingSteps, size_t rangeMin, size_t rangeMax) = 0;
	virtual void	GatherSlopes(float *dst, const CLayer *prevLayer, size_t rangeMin, size_t rangeMax) const = 0;
	virtual void	PrintInfo() const = 0;
	virtual void	Serialize(std::vector<uint8_t> &data) const = 0;
	virtual bool	UnSerialize(const std::vector<uint8_t> &data, size_t &curIdx) = 0;

	virtual size_t	GetThreadingHint() const = 0;
	virtual size_t	GetDomainSize() const = 0;

	void			SetActivation(EActivation activation) { m_Activation = activation; }
	void			SetInitialization(ERandInitializer initializer) { m_Initializer = initializer; }
	void			SetOptimizaton(EOptimization optimizer) { m_Optimization = optimizer; }
	void			SetRegularization(ERegularizer regularizer) { m_Regularizer = regularizer; }
	void			SetLearningRate(float learningRate) { m_LearningRate = learningRate; }
	void			SetLearn(bool learn) { m_Learn = learn; }
	bool			Learn() const { return m_Learn; }

	void			Initializer();

protected:
	void			SerializeLayerType(std::vector<uint8_t> &data, ELayerType type) const;
	void			SerializeInOutSize(std::vector<uint8_t> &data) const;
	bool			UnSerializeInOutSize(const std::vector<uint8_t> &data, size_t &curIdx);
	void			SerializeBasicInfo(std::vector<uint8_t> &data) const;
	bool			UnSerializeBasicInfo(const std::vector<uint8_t> &data, size_t &curIdx);
	void			SerializeWeightsAndBias(std::vector<uint8_t> &data) const;
	bool			UnSerializeWeightsAndBias(const std::vector<uint8_t> &data, size_t &curIdx);

	void			PrintBasicInfo() const;
	void			InitializeRandomRange(float min, float max);

	// Activations:
	void		Activation(float *dst, const float *src, size_t size) const;
	void		ActivationDerivative(float* dst, const float* src, size_t size) const;

	float		Sigmoid(float x) const;
	float		SigmoidDerivative(float x) const;
	float		Relu(float x) const;
	float		ReluDerivative(float x) const;
	float		LeakyRelu(float x) const;
	float		LeakyReluDerivative(float x) const;
	float		Elu(float x) const;
	float		EluDerivative(float x) const;
	float		Gelu(float x) const;
	float		GeluDerivative(float x) const;
	float		Tanh(float x) const;
	float		TanhDerivative(float x) const;

	// Optimization:
	void		OptimizeWeight(size_t rangeMin, size_t rangeMax, size_t trainingSteps);
	void		OptimizeBias(float *biases, const float *deltas, size_t minRange, size_t maxRange, size_t trainingSteps);

	void		SGDWeight(size_t rangeMin, size_t rangeMax, size_t trainingSteps);
	void		SGDBias(float *biases, const float *deltas, size_t minRange, size_t maxRange, size_t trainingSteps);

	void		AdagradWeight(size_t rangeMin, size_t rangeMax, size_t trainingSteps);
	void		AdagradBias(float* biases, const float* deltas, size_t minRange, size_t maxRange, size_t trainingSteps);

	CNeuronMatrix		m_Weights;
	CNeuronVector		m_Bias;

	size_t				m_InputSize;
	size_t				m_OutputSize;
	CNeuronVector		m_NetInput;
	CNeuronVector		m_Output;
	CNeuronVector		m_SlopesOut;
	CNeuronVector		m_SlopesOutAccum;
	CNeuronMatrix		m_SlopesWeightAccum;
	CNeuronMatrix		m_AdagradWeightAccum;
	CNeuronVector		m_AdagradBiasAccum;
	
	EActivation			m_Activation;
	EOptimization		m_Optimization;
	ERandInitializer	m_Initializer;
	ERegularizer		m_Regularizer;
	float				m_RegularizerRatio;

	float				m_LearningRate;
	float				m_Inertia;

	CNeuronMatrix		m_DeltaWeightVelocity;
	CNeuronVector		m_DeltaBiasVelocity;

	bool				m_Learn;

	struct	SSerializedLayerBasicInfo
	{
		uint32_t	m_Activation;
		uint32_t	m_Optimization;
		uint32_t	m_Initializer;
		uint32_t	m_Regularizer;
		float		m_RegularizerRatio;
		float		m_LearningRate;
		float		m_Inertia;
	};
};

