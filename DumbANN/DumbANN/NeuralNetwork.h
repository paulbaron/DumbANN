#pragma once

#include "LayerBase.h"
#include "TaskManager.h"

#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <thread>

class	CNeuralNetwork
{
public:
	CNeuralNetwork();
	~CNeuralNetwork();

	bool	AddLayer(CLayer *layer);
	bool	FeedForward(const float *input);
	bool	BackPropagateError(const float *input, const std::vector<float> &expected);
	bool	UpdateWeightAndBiases();

	const CNeuronVector		&GetOutput() const { return m_Layers.back()->GetOutput(); }
	void					DestroyThreadsIFN() { m_TaskManager.DestroyThreadsIFN(); }

	void	PrintDetails() const;

private:
	void	ResetTrainingSteps() { m_CurrentTrainingStep = 0; }

	std::vector<CLayer*>		m_Layers;
	uint32_t					m_CurrentTrainingStep;

	CTaskManager				m_TaskManager;
};
