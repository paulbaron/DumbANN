
#include "LayerBase.h"
#include <xmmintrin.h>
#include <assert.h>

#include "LayerConv2D.h"
#include "LayerDense.h"
#include "LayerDropout.h"
#include "LayerMaxPooling.h"
#include "LayerSoftmax.h"

float	RemapValue(float value, float oldMin, float oldMax, float newMin, float newMax)
{
	const float		normalizedVaue = (value - oldMin) / (oldMax - oldMin);
	return normalizedVaue * (newMax - newMin) + newMin;
}

const char	*kActivationNames[]
{
	"Relu",
	"LeakyRelu",
	"Elu",
	"Gelu",
	"Sigmoid",
	"Tanh",
	"Linear"
};

const char	*kOptimizationNames[]
{
	"SGD",
	"Adagrad"
};

const char	*kRegularizationNames[]
{
	"None",
	"L1",
	"L2"
};

const char	*kInitializerNames[]
{
	"Rand Uniform [0, 1]",
	"Rand Uniform [-1, 1]",
	"Rand Xavier",
	"Rand Xavier Normalized",
	"Rand He"
};

CLayer	*CLayer::CreateLayer(ELayerType type)
{
	switch (type)
	{
	case ELayerType::LayerDense:
		return new CLayerDense();
		break;
	case ELayerType::LayerConv2D:
		return new CLayerConv2D();
		break;
	case ELayerType::LayerMaxPooling:
		return new CLayerMaxPooling2D();
		break;
	case ELayerType::LayerDropout:
		return new CLayerDropOut();
		break;
	case ELayerType::LayerSofmax:
		return new CLayerSoftMax();
		break;
	default:
		return nullptr;
		break;
	}
}

CLayer::CLayer()
:	m_InputSize(0)
,	m_OutputSize(0)
,	m_Activation(EActivation::Sigmoid)
,	m_Optimization(EOptimization::Adagrad)
,	m_Initializer(ERandInitializer::RandXavier)
,	m_Regularizer(ERegularizer::None)
,	m_RegularizerRatio(1e-5)
,	m_LearningRate(0.001f)
,	m_Inertia(0.0f)
,	m_Learn(true)
{
}

CLayer::~CLayer()
{
}

void	CLayer::Initializer()
{
	memset(m_Weights.Data(), 0, m_Weights.StorageByteSize());
	memset(m_Bias.Data(), 0, m_Bias.Size() * sizeof(float));
	memset(m_DeltaWeightVelocity.Data(), 0, m_DeltaWeightVelocity.StorageByteSize());
	memset(m_DeltaBiasVelocity.Data(), 0, m_DeltaBiasVelocity.Size() * sizeof(float));

	if (m_Initializer == ERandInitializer::RandUniform_0_1)
		InitializeRandomRange(0, 1);
	else if (m_Initializer == ERandInitializer::RandUniform_MINUS1_1)
		InitializeRandomRange(-1, 1);
	else if (m_Initializer == ERandInitializer::RandXavier)
	{
		float	n = static_cast<float>(m_InputSize);
		float	range = 1.0f / static_cast<float>(sqrt(n));
		InitializeRandomRange(-range, range);
	}
	else if (m_Initializer == ERandInitializer::RandXavierNormalized)
	{
		float	n = static_cast<float>(m_InputSize);
		float	m = static_cast<float>(m_Output.Size());
		float	range = static_cast<float>(sqrt(6.0f)) / static_cast<float>(sqrt(n + m));
		InitializeRandomRange(-range, range);
	}
	else if (m_Initializer == ERandInitializer::RandHe)
	{
		float	n = static_cast<float>(m_InputSize);
		float	range = sqrt(2.0f / n);
		InitializeRandomRange(-range, range);
	}
	else
	{
		assert(false);
	}
}

void	CLayer::PrintBasicInfo() const
{
	printf("\t\tActivation: %s\n", kActivationNames[(int)m_Activation]);
	printf(	"\t\tOptimization: %s (regularize: %s)\n",
			kOptimizationNames[(int)m_Optimization],
			kRegularizationNames[(int)m_Regularizer]);
	printf("\t\tWeight Initializer: %s\n", kInitializerNames[(int)m_Initializer]);
}

void	CLayer::SerializeLayerType(std::vector<uint8_t> &data, ELayerType type) const
{
	size_t		prevSize = data.size();
	data.resize(prevSize + sizeof(uint32_t));
	*(uint32_t*)(data.data() + prevSize) = (uint32_t)type;
}

void	CLayer::SerializeInOutSize(std::vector<uint8_t> &data) const
{
	size_t		prevSize = data.size();
	data.resize(prevSize + 2 * sizeof(uint32_t));
	uint32_t	*dataPtr = (uint32_t*)(data.data() + prevSize);
	dataPtr[0] = m_InputSize;
	dataPtr[1] = m_OutputSize;
}

bool	CLayer::UnSerializeInOutSize(const std::vector<uint8_t> &data, size_t &curIdx)
{
	if (curIdx + 2 * sizeof(uint32_t) > data.size())
		return false;
	uint32_t	*dataPtr = (uint32_t*)(data.data() + curIdx);
	m_InputSize = dataPtr[0];
	m_OutputSize = dataPtr[1];
	curIdx += 2 * sizeof(uint32_t);
	return true;
}

void	CLayer::SerializeBasicInfo(std::vector<uint8_t> &data) const
{
	SSerializedLayerBasicInfo	info;
	info.m_Activation = (uint32_t)m_Activation;
	info.m_Optimization = (uint32_t)m_Optimization;
	info.m_Initializer = (uint32_t)m_Initializer;
	info.m_Regularizer = (uint32_t)m_Regularizer;
	info.m_RegularizerRatio = m_RegularizerRatio;
	info.m_LearningRate = m_LearningRate;
	info.m_Inertia = m_Inertia;
	size_t	prevSize = data.size();
	data.resize(prevSize + sizeof(info));
	*(SSerializedLayerBasicInfo*)(data.data() + prevSize) = info;
}

bool	CLayer::UnSerializeBasicInfo(const std::vector<uint8_t> &data, size_t &curIdx)
{
	if (curIdx + sizeof(SSerializedLayerBasicInfo) > data.size())
		return false;
	const SSerializedLayerBasicInfo	*info = (const SSerializedLayerBasicInfo*)(data.data() + curIdx);
	m_Activation = (EActivation)info->m_Activation;
	m_Optimization = (EOptimization)info->m_Optimization;
	m_Initializer = (ERandInitializer)info->m_Initializer;
	m_Regularizer = (ERegularizer)info->m_Regularizer;
	m_RegularizerRatio = info->m_RegularizerRatio;
	m_LearningRate = info->m_LearningRate;
	m_Inertia = info->m_Inertia;
	curIdx += sizeof(SSerializedLayerBasicInfo);
	return true;
}

void	CLayer::SerializeWeightsAndBias(std::vector<uint8_t> &data) const
{
	m_Weights.Serialize(data);
	m_Bias.Serialize(data);
}

bool	CLayer::UnSerializeWeightsAndBias(const std::vector<uint8_t> &data, size_t &curIdx)
{
	if (!m_Weights.UnSerialize(data, curIdx))
		return false;
	m_Weights.DebugCheckForNaNs();
	if (!m_Bias.UnSerialize(data, curIdx))
		return false;
	m_Bias.DebugCheckForNaNs();
	return true;
}

void	CLayer::InitializeRandomRange(float min, float max)
{
	// Initialize to random floats:
	const SNeuronMatrixView& view = m_Weights.View();
	for (size_t rowIdx = 0; rowIdx < view.m_Rows; ++rowIdx)
	{
		float* rowPtr = view.GetRow(rowIdx);
		for (size_t colIdx = 0; colIdx < view.m_Columns; ++colIdx)
		{
			float	randf = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
			rowPtr[colIdx] = randf * (max - min) + min;
		}
	}
}

void	CLayer::Activation(float* dst, const float* src, size_t size) const
{
	switch (m_Activation)
	{
	case EActivation::Relu:
		for (int i = 0; i < size; ++i)
			dst[i] = Relu(src[i]);
		break;
	case EActivation::LeakyRelu:
		for (int i = 0; i < size; ++i)
			dst[i] = LeakyRelu(src[i]);
		break;
	case EActivation::Gelu:
		for (int i = 0; i < size; ++i)
			dst[i] = Gelu(src[i]);
		break;
	case EActivation::Sigmoid:
		for (int i = 0; i < size; ++i)
			dst[i] = Sigmoid(src[i]);
		break;
	case EActivation::Tanh:
		for (int i = 0; i < size; ++i)
			dst[i] = Tanh(src[i]);
		break;
	case EActivation::Linear:
	default:
		for (int i = 0; i < size; ++i)
			dst[i] = src[i];
		break;
	}
}

void	CLayer::ActivationDerivative(float* dst, const float* src, size_t size) const
{
	switch (m_Activation)
	{
	case EActivation::Relu:
		for (int i = 0; i < size; ++i)
			dst[i] *= ReluDerivative(src[i]);
		break;
	case EActivation::LeakyRelu:
		for (int i = 0; i < size; ++i)
			dst[i] *= LeakyReluDerivative(src[i]);
		break;
	case EActivation::Gelu:
		for (int i = 0; i < size; ++i)
			dst[i] *= GeluDerivative(src[i]);
		break;
	case EActivation::Sigmoid:
		for (int i = 0; i < size; ++i)
			dst[i] *= SigmoidDerivative(src[i]);
		break;
	case EActivation::Tanh:
		for (int i = 0; i < size; ++i)
			dst[i] *= TanhDerivative(src[i]);
		break;
	case EActivation::Linear:
		// dst *= 1
	default:
		break;
	}
}

float	CLayer::Sigmoid(float x) const
{
	return 1.0f / (1.0f + exp(-x));
}

float	CLayer::SigmoidDerivative(float x) const
{
	return Sigmoid(x) * (1.0f - Sigmoid(x));
}

float	CLayer::Relu(float x) const
{
	return x > 0 ? x : 0;
}

float	CLayer::ReluDerivative(float x) const
{
	return x >= 0 ? 1 : 0;
}

float	CLayer::LeakyRelu(float x) const
{
	return x > 0 ? x : 0.001f * x;
}

float	CLayer::LeakyReluDerivative(float x) const
{
	return x >= 0 ? 1 : 0.001f;
}

float	CLayer::Elu(float x) const
{
	return x >= 0 ? 1 : (expf(x) - 1);
}

float	CLayer::EluDerivative(float x) const
{
	return x >= 0 ? 1 : expf(x);
}

float	CLayer::Gelu(float x) const
{
	const float	a = 0.0356774f * (x * x * x) + 0.797885f * x;
	return 0.5f * x * (tanhf(a) + 1.0f);
}

float	CLayer::GeluDerivative(float x) const
{
	const float	cubeX = x * x * x;
	const float	a = 0.0356774f * cubeX + 0.797885f * x;
	const float coshA = coshf(a);
	const float cosh2A = coshf(2 * a);
	const float coshASquare = coshA * coshA;
	const float b = (cosh2A + 1) * (cosh2A + 1);

	return	(0.214064f * cubeX * coshASquare) / b +
			(1.59577f * x * coshASquare) / b +
			(0.5f * sinhf(2 * a)) / (cosh2A + 1) + 0.5f;
}

float	CLayer::Tanh(float x) const
{
	return tanhf(x);
}

float	CLayer::TanhDerivative(float x) const
{
	float	thx = tanhf(x);
	return 1.0f - (thx * thx);
}

void	CLayer::OptimizeWeight(size_t rangeMin, size_t rangeMax, size_t trainingSteps)
{
	if (m_Optimization == EOptimization::SGD)
		SGDWeight(rangeMin, rangeMax, trainingSteps);
	else if (m_Optimization == EOptimization::Adagrad)
		AdagradWeight(rangeMin, rangeMax, trainingSteps);
	else
	{
		assert(false);
		SGDWeight(rangeMin, rangeMax, trainingSteps);
	}
}

void	CLayer::OptimizeBias(float* biases, const float* deltas, size_t minRange, size_t maxRange, size_t trainingSteps)
{
	if (m_Optimization == EOptimization::SGD)
		SGDBias(biases, deltas, minRange, maxRange, trainingSteps);
	else if (m_Optimization == EOptimization::Adagrad)
		AdagradBias(biases, deltas, minRange, maxRange, trainingSteps);
	else
	{
		assert(false);
		SGDBias(biases, deltas, minRange, maxRange, trainingSteps);
	}
}

void	CLayer::SGDWeight(size_t rangeMin, size_t rangeMax, size_t trainingSteps)
{
#if 0
	// Reference non-SIMD code:
	for (size_t y = rangeMin; y < rangeMax; ++y)
	{
		for (size_t x = 0; x < m_Weights.View().m_Columns; ++x)
		{
			float	avgDelta = m_SlopesWeightAccum.View().GetRow(y)[x] / static_cast<float>(trainingSteps);
			m_DeltaWeightVelocity.View().GetRow(y)[x] = m_DeltaWeightVelocity.View().GetRow(y)[x] * m_Inertia + m_LearningRate * avgDelta;
			m_Weights.View().GetRow(y)[x] -= m_DeltaWeightVelocity.View().GetRow(y)[x];
		}
	}
	return;
#endif
	float			tSteps = static_cast<float>(trainingSteps);
	const __m128	invTrainSteps_xxxx = _mm_set1_ps(1.0f / tSteps);
	const __m128	inertia_xxxx = _mm_set1_ps(m_Inertia);
	const __m128	learningRate_xxxx = _mm_set1_ps(m_LearningRate);
	float			*deltaWeightVelocityPtr = m_DeltaWeightVelocity.View().GetRow(rangeMin);
	const float		*deltasPtr = m_SlopesWeightAccum.View().GetRow(rangeMin);
	float			*weightsPtr = m_Weights.View().GetRow(rangeMin);
	const float		*weightsPtrStop = m_Weights.View().GetRow(rangeMax);

	// Contiguous matrix:
	assert(m_Weights.View().m_RowByteStride - (m_Weights.View().m_Columns * 4) < 0x10);
	// Aligned pointers:
	assert(	((ptrdiff_t)deltasPtr & 0xF) == 0 &&
			((ptrdiff_t)weightsPtr & 0xF) == 0 &&
			((ptrdiff_t)deltaWeightVelocityPtr & 0xF) == 0);
	if (m_Regularizer == ERegularizer::None)
	{
		while (weightsPtr < weightsPtrStop)
		{
			const __m128	weights_xyzw = _mm_load_ps(weightsPtr);
			const __m128	deltas_xyzw = _mm_load_ps(deltasPtr);
			const __m128	velocities_xyzw = _mm_load_ps(deltaWeightVelocityPtr);
			const __m128	deltaFinal_xyzw = _mm_mul_ps(learningRate_xxxx, _mm_mul_ps(deltas_xyzw, invTrainSteps_xxxx));
			const __m128	velocityFinal_xyzw = _mm_sub_ps(_mm_mul_ps(inertia_xxxx, velocities_xyzw), deltaFinal_xyzw);
			const __m128	weightFinal = _mm_add_ps(weights_xyzw, velocityFinal_xyzw);

			_mm_store_ps(deltaWeightVelocityPtr, velocityFinal_xyzw);
			_mm_store_ps(weightsPtr, weightFinal);
			deltaWeightVelocityPtr += 4;
			weightsPtr += 4;
			deltasPtr += 4;
		}
	}
	else if (m_Regularizer == ERegularizer::L1)
	{
		const __m128	zero = _mm_set_ps1(0.0f);
		const __m128	neg1 = _mm_set_ps1(-1.0f);
		const __m128	pos1 = _mm_set_ps1(1.0f);
		const __m128	regularizerRatio_xxxx = _mm_set1_ps(m_RegularizerRatio);
		const __m128	regularizerSlope_xxxx = _mm_mul_ps(_mm_mul_ps(regularizerRatio_xxxx, learningRate_xxxx), invTrainSteps_xxxx);

		while (weightsPtr < weightsPtrStop)
		{
			const __m128	weights_xyzw = _mm_load_ps(weightsPtr);
			const __m128	deltas_xyzw = _mm_load_ps(deltasPtr);
			const __m128	velocities_xyzw = _mm_load_ps(deltaWeightVelocityPtr);
			const __m128	deltaFinal_xyzw = _mm_mul_ps(learningRate_xxxx, _mm_mul_ps(deltas_xyzw, invTrainSteps_xxxx));
			const __m128	velocityFinal_xyzw = _mm_sub_ps(_mm_mul_ps(inertia_xxxx, velocities_xyzw), deltaFinal_xyzw);
			const __m128	positives_xyz = _mm_and_ps(_mm_cmpgt_ps(weights_xyzw, zero), pos1);
			const __m128	negatives_xyz = _mm_and_ps(_mm_cmplt_ps(weights_xyzw, zero), neg1);
			const __m128	weightSign_xyz = _mm_or_ps(positives_xyz, negatives_xyz);
			const __m128	regSlope_xyz = _mm_mul_ps(weightSign_xyz, regularizerSlope_xxxx);
			const __m128	regWeight_xyz = _mm_sub_ps(weights_xyzw, regSlope_xyz);
			const __m128	weightFinal = _mm_add_ps(regWeight_xyz, velocityFinal_xyzw);

			_mm_store_ps(deltaWeightVelocityPtr, velocityFinal_xyzw);
			_mm_store_ps(weightsPtr, weightFinal);
			deltaWeightVelocityPtr += 4;
			weightsPtr += 4;
			deltasPtr += 4;
		}
	}
	else if (m_Regularizer == ERegularizer::L2)
	{
		const __m128	regularizerRatio_xxxx = _mm_set1_ps(m_RegularizerRatio);
		const __m128	regularizerSlope_xxxx = _mm_mul_ps(_mm_mul_ps(regularizerRatio_xxxx, learningRate_xxxx), invTrainSteps_xxxx);

		while (weightsPtr < weightsPtrStop)
		{
			const __m128	weights_xyzw = _mm_load_ps(weightsPtr);
			const __m128	deltas_xyzw = _mm_load_ps(deltasPtr);
			const __m128	velocities_xyzw = _mm_load_ps(deltaWeightVelocityPtr);
			const __m128	deltaFinal_xyzw = _mm_mul_ps(learningRate_xxxx, _mm_mul_ps(deltas_xyzw, invTrainSteps_xxxx));
			const __m128	velocityFinal_xyzw = _mm_sub_ps(_mm_mul_ps(inertia_xxxx, velocities_xyzw), deltaFinal_xyzw);
			const __m128	regSlope_xyz = _mm_mul_ps(weights_xyzw, regularizerSlope_xxxx);
			const __m128	regWeight_xyz = _mm_sub_ps(weights_xyzw, regSlope_xyz);
			const __m128	weightFinal = _mm_add_ps(regWeight_xyz, velocityFinal_xyzw);

			_mm_store_ps(deltaWeightVelocityPtr, velocityFinal_xyzw);
			_mm_store_ps(weightsPtr, weightFinal);
			deltaWeightVelocityPtr += 4;
			weightsPtr += 4;
			deltasPtr += 4;
		}
	}
}

void	CLayer::SGDBias(float* biases, const float* deltas, size_t minRange, size_t maxRange, size_t trainingSteps)
{
	float* deltaBiasVelocityPtr = m_DeltaBiasVelocity.Data();
	for (size_t i = minRange; i < maxRange; ++i)
	{
		const float	averageDelta = deltas[i] / static_cast<float>(trainingSteps);
		deltaBiasVelocityPtr[i] = m_Inertia * deltaBiasVelocityPtr[i] - m_LearningRate * averageDelta;
		biases[i] += deltaBiasVelocityPtr[i];
	}
}

void	CLayer::AdagradWeight(size_t rangeMin, size_t rangeMax, size_t trainingSteps)
{
	const float		*deltasPtr = m_SlopesWeightAccum.View().GetRow(rangeMin);
	float			*weightsPtr = m_Weights.View().GetRow(rangeMin);
	const float		*weightsPtrStop = m_Weights.View().GetRow(rangeMax);
	float			*adagradAccumPtr = m_AdagradWeightAccum.View().GetRow(rangeMin);
	const float		epsilon = 0.00001f;

	while (weightsPtr < weightsPtrStop)
	{
		const float		avgDelta = m_LearningRate * (*deltasPtr / static_cast<float>(trainingSteps));
		*adagradAccumPtr += avgDelta * avgDelta;
		const float		adjustedDelta = avgDelta / (epsilon + sqrt(*adagradAccumPtr));
		*weightsPtr -= adjustedDelta;
		++deltasPtr;
		++weightsPtr;
		++adagradAccumPtr;
	}
}

void	CLayer::AdagradBias(float* biases, const float* deltas, size_t minRange, size_t maxRange, size_t trainingSteps)
{
	float			*adagradAccumPtr = m_AdagradBiasAccum.Data();
	const float		epsilon = 0.00001f;

	for (size_t i = minRange; i < maxRange; ++i)
	{
		const float		avgDelta = m_LearningRate * (deltas[i] / static_cast<float>(trainingSteps));
		adagradAccumPtr[i] += avgDelta * avgDelta;
		const float		adjustedDelta = avgDelta / (epsilon + sqrt(adagradAccumPtr[i]));
		biases[i] -= adjustedDelta;
	}
}
