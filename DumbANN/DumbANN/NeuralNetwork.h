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
	bool	BackPropagateError(const float *input, const float *expected);
	bool	UpdateWeightAndBiases();

	const CNeuronVector		&GetOutput() const { return m_Layers.back()->GetOutput(); }
	void					DestroyThreadsIFN() { m_TaskManager.DestroyThreadsIFN(); }

	const std::vector<CLayer*>	&Layers() const { return m_Layers; }

	void	PrintDetails() const;

	bool	Serialize(const char *path);
	bool	UnSerialize(const char *path);

	void	SetAllLearningRate(float learningRate);

private:
	void	ResetTrainingSteps() { m_CurrentTrainingStep = 0; }

	std::vector<CLayer*>		m_Layers;
	uint32_t					m_CurrentTrainingStep;

	CTaskManager				m_TaskManager;

	// Serializer:
	struct	SNetworkHeader
	{
		static const uint32_t	MagicNumber = 0x0D04BA44;
		uint32_t	m_Magic = 0;
		uint32_t	m_LayerCount = 0;
	};
};
