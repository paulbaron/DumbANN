
#include "NeuralNetwork.h"

#include <xmmintrin.h>

#include <assert.h>

#include <thread>
#include <algorithm>

CNeuralNetwork::CNeuralNetwork()
:	m_CurrentTrainingStep(0)
{
}

CNeuralNetwork::~CNeuralNetwork()
{
}

bool	CNeuralNetwork::AddLayer(CLayer *layer)
{
	if (!m_Layers.empty())
	{
		if (m_Layers.back()->GetOutputSize() != layer->GetInputSize())
			return false;
	}
	m_Layers.push_back(layer);
	return true;
}

bool	CNeuralNetwork::FeedForward(const float *input)
{
	if (!m_Layers.empty())
	{
		for (size_t i = 0; i < m_Layers.size(); ++i)
		{
			const float					*nextInput = (i == 0) ? input : m_Layers[i - 1]->GetOutput().Data();
			CLayer						*layer = m_Layers[i];
			std::function<void(size_t, size_t)>	feedForward = [layer, nextInput](size_t minRange, size_t maxRange)
			{
				layer->FeedForward(nextInput, minRange, maxRange);
			};
			// Feed forward is FAST, we can reduce the threading hint:
			m_TaskManager.MultithreadRange(feedForward, layer->GetDomainSize(), layer->GetThreadingHint() / 8);
		}
	}
	return true;
}

bool	CNeuralNetwork::BackPropagateError(const float *input, const std::vector<float> &expected)
{
	if (!m_Layers.empty())
	{
		assert(m_Layers.back()->GetOutputSize() == expected.size());

		std::vector<float>			error;
		const float					*output = m_Layers.back()->GetOutput().Data();

		error.resize(expected.size());
		for (int i = 0; i < error.size(); ++i)
			error[i] = expected[i] - output[i];

		for (int i = m_Layers.size() - 1; i >= 0; --i)
		{
			CLayer			*layer = m_Layers[i];
			const float		*prevOutput = (i == 0) ? input : m_Layers[i - 1]->GetOutput().Data();
			const CLayer	*nextLayer = (i == m_Layers.size() - 1) ? nullptr : m_Layers[i + 1];

			std::function<void(size_t, size_t)>	backProp = [layer, prevOutput, nextLayer, &error](size_t minRange, size_t maxRange)
			{
				if (nextLayer == nullptr)
					layer->BackPropagateError(prevOutput, error, minRange, maxRange);
				else
					layer->BackPropagateError(prevOutput, nextLayer, minRange, maxRange);
			};
			m_TaskManager.MultithreadRange(backProp, layer->GetDomainSize(), layer->GetThreadingHint());
		}
	}
	++m_CurrentTrainingStep;
	return true;
}

bool	CNeuralNetwork::UpdateWeightAndBiases()
{
	for (int i = 0; i < m_Layers.size(); ++i)
	{
		CLayer	*layer = m_Layers[i];

		std::function<void(size_t, size_t)>	updateWeightAndBias = [this, layer](size_t minRange, size_t maxRange)
		{
			layer->UpdateWeightsAndBias(m_CurrentTrainingStep, minRange, maxRange);
		};
		m_TaskManager.MultithreadRange(updateWeightAndBias, layer->GetDomainSize(), layer->GetThreadingHint(), false);
	}
	m_TaskManager.CallOnceJobFinished(std::function<void()>([this](){ ResetTrainingSteps(); }));
	return true;
}
