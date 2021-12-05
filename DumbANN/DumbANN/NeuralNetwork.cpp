
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

		if (layer->Learn())
		{
			std::function<void(size_t, size_t)>	updateWeightAndBias = [this, layer](size_t minRange, size_t maxRange)
			{
				layer->UpdateWeightsAndBias(m_CurrentTrainingStep, minRange, maxRange);
			};
			m_TaskManager.MultithreadRange(updateWeightAndBias, layer->GetDomainSize(), layer->GetThreadingHint(), false);
		}
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

bool	CNeuralNetwork::Serialize(const char* path)
{
	m_TaskManager.WaitForCompletion(true);
	FILE	*annFile = nullptr;
	if (fopen_s(&annFile, path, "w+b") != 0)
	{
		fprintf(stderr, "Could not open ann file\n");
		return false;
	}
	SNetworkHeader	header;
	header.m_Magic = SNetworkHeader::MagicNumber;
	header.m_LayerCount = m_Layers.size();
	fwrite(&header, sizeof(SNetworkHeader), 1, annFile);
	std::vector<uint8_t>	writeBuff;
	for (const CLayer *layer : m_Layers)
	{
		layer->Serialize(writeBuff);
	}
	printf("Writing file '%s'\n", path);
	fwrite(writeBuff.data(), sizeof(uint8_t), writeBuff.size(), annFile);
	fclose(annFile);
	return true;
}

bool	CNeuralNetwork::UnSerialize(const char* path)
{
	FILE	*annFile = nullptr;
	if (fopen_s(&annFile, path, "rb") != 0)
	{
		fprintf(stderr, "Could not open ann file\n");
		return false;
	}
	SNetworkHeader	header;
	fread(&header, sizeof(SNetworkHeader), 1, annFile);

	if (header.m_Magic != SNetworkHeader::MagicNumber)
	{
		fprintf(stderr, "Wrong magic number\n");
		fclose(annFile);
		return false;
	}

	std::vector<uint8_t>	readBuff;

	long	cursorPos = ftell(annFile);
	fseek(annFile, 0, SEEK_END);
	long	cursorEnd = ftell(annFile);
	fseek(annFile, cursorPos, SEEK_SET);
	readBuff.resize(cursorEnd - cursorPos);
	fread(readBuff.data(), sizeof(uint8_t), readBuff.size(), annFile);

	size_t		curIdx = 0;

	while (curIdx < readBuff.size())
	{
		uint32_t	*dataPtr = (uint32_t*)(readBuff.data() + curIdx);
		curIdx += sizeof(uint32_t);
		CLayer		*layer =  CLayer::CreateLayer((ELayerType)*dataPtr);
		if (layer == nullptr)
		{
			fclose(annFile);
			return false;
		}
		if (!layer->UnSerialize(readBuff, curIdx))
		{
			fprintf(stderr, "Failed unserializing layer\n");
			fclose(annFile);
			return false;
		}
		if (!AddLayer(layer))
		{
			fclose(annFile);
			return false;
		}
	}
	fclose(annFile);
}

void	CNeuralNetwork::SetAllLearningRate(float learningRate)
{
	for (CLayer *layer : m_Layers)
		layer->SetLearningRate(learningRate);
}

