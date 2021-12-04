
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
		assert(m_Layers.back()->GetOutputSize() == layer->GetInputSize());
		if (m_Layers.back()->GetOutputSize() != layer->GetInputSize())
			return false;
	}
	m_Layers.push_back(layer);
	return true;
}

bool	CNeuralNetwork::FeedForward(const float *input)
{
	MICROPROFILE_SCOPEI("CNeuralNetwork", "FeedForward", MP_GREEN3);
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

bool	CNeuralNetwork::BackPropagateError(const float *input, const float *expected)
{
	MICROPROFILE_SCOPEI("CNeuralNetwork", "BackPropagateError", MP_RED3);
	if (!m_Layers.empty())
	{
		std::vector<float>			error;
		size_t						outSize = m_Layers.back()->GetOutputSize();
		const float					*output = m_Layers.back()->GetOutput().Data();

		error.resize(outSize);
		for (int i = 0; i < error.size(); ++i)
			error[i] = expected[i] - output[i];

		for (int i = m_Layers.size() - 1; i >= 0; --i)
		{
			CLayer			*layer = m_Layers[i];
			const CLayer	*nextLayer = (i == m_Layers.size() - 1) ? nullptr : m_Layers[i + 1];
			const CLayer	*prevLayer = (i == 0) ? nullptr : m_Layers[i - 1];
			const float		*prevOutput = (prevLayer == nullptr) ? input : prevLayer->GetOutput().Data();

			std::function<void(size_t, size_t)>	backProp = [&](size_t minRange, size_t maxRange)
			{
				if (nextLayer == nullptr)
					layer->BackPropagateError(prevOutput, error, minRange, maxRange);
				else
				{
					layer->BackPropagateError(prevOutput, nextLayer, minRange, maxRange);
				}
			};
			m_TaskManager.MultithreadRange(backProp, layer->GetDomainSize(), layer->GetThreadingHint());
			if (prevLayer != nullptr)
			{
				std::function<void(size_t, size_t)>	gatherSlopes = [&](size_t minRange, size_t maxRange)
				{
					layer->GatherSlopes(prevLayer->GetSlopesOut().Data(),
										prevLayer,
										minRange, maxRange);
				};
				// Can be expensive, ThreadHint * 8 to split in more tasks:
				m_TaskManager.MultithreadRange(	gatherSlopes,
												prevLayer->GetSlopesOut().Size(),
												prevLayer->GetSlopesOut().Size() * 8);
			}
		}
	}
	++m_CurrentTrainingStep;
	return true;
}

bool	CNeuralNetwork::UpdateWeightAndBiases()
{
	MICROPROFILE_SCOPEI("CNeuralNetwork", "UpdateWeightAndBiases", MP_BLUE3);
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

void	CNeuralNetwork::PrintDetails() const
{
	printf("-------------------------------\n");
	printf("Neural Network with %zu layers:\n", m_Layers.size());
	for (const CLayer *layer : m_Layers)
	{
		printf("-------------------------------\n");
		layer->PrintInfo();
	}
	printf("-------------------------------\n");
}
