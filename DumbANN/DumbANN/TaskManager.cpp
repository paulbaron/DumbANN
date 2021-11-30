#include "TaskManager.h"

#include <algorithm>
#include <assert.h>

#define BATCH_TASK_COUNT				32
#define MIN_VALUES_COMPUTED_PER_TASK	8192

CTaskManager::CTaskManager()
:	m_ToComplete(0)
,	m_OnceJobFinished(nullptr)
,	m_StopThreads(false)
{
}

CTaskManager::~CTaskManager()
{
	DestroyThreadsIFN();
}

void	CTaskManager::MultithreadRange(std::function<void(size_t, size_t)> function, size_t outSize, size_t domainSize, bool sync)
{
	size_t		taskCount = std::min(domainSize / MIN_VALUES_COMPUTED_PER_TASK, m_Threads.size() * 2);

	CreateThreadsIFN();
	if (sync)
		WaitForCompletion(true);
	if (taskCount <= 1)
	{
		function(0, outSize); // Not worth multi-threading
	}
	else
	{
		if (taskCount > outSize)
			taskCount = outSize;
		std::function<void()>	functions[BATCH_TASK_COUNT] = { };
		size_t					taskIdx = 0;
		size_t					taskIdxInBatch = 0;

		while (taskIdx < taskCount)
		{
			size_t					taskIdxInBatch = 0;

			while (taskIdx < taskCount && taskIdxInBatch < BATCH_TASK_COUNT)
			{
				size_t	range = outSize / taskCount;
				size_t	minRange = taskIdx * range;
				size_t	maxRange = ((taskIdx + 1) < taskCount) ? minRange + range : outSize;

				functions[taskIdxInBatch] = std::function<void()>([minRange, maxRange, function]()
				{
					function(minRange, maxRange);
				});
				++taskIdx;
				++taskIdxInBatch;
			}
			{
				m_QueueLock.lock();
				for (int i = 0; i < taskIdxInBatch; ++i)
					m_Queue.push(functions[i]);
				m_ToComplete += taskIdxInBatch;
				m_QueueLock.unlock();
				m_QueueChanged.notify_all();
			}
		}
		if (sync)
			WaitForCompletion(true);
	}
}

void	CTaskManager::CreateThreadsIFN(int count)
{
	if (m_Threads.empty() && count != 0)
	{
		size_t	processorCount = (size_t)count;
		if (count < 0)
		{
			processorCount = std::thread::hardware_concurrency();
			processorCount = processorCount <= 1 ? 1 : processorCount - 1;
		}
		m_Threads.resize(processorCount);
		for (int i = 0; i < m_Threads.size(); ++i)
		{
			m_Threads[i] = new std::thread([this]()
			{
				ConsumerThreadUpdate();
			});
		}
	}
}

void	CTaskManager::DestroyThreadsIFN()
{
	m_StopThreads = true;
	m_QueueChanged.notify_all();
	for (int i = 0; i < m_Threads.size(); ++i)
	{
		m_Threads[i]->join();
		delete m_Threads[i];
	}
	m_Threads.clear();
}

void	CTaskManager::WaitForCompletion(bool processTasks)
{
	if (processTasks)
	{
		std::function<void()>		toExec = nullptr;
		bool						waitForThreads = true;
		while (waitForThreads)
		{
			{
				std::unique_lock<std::mutex>	queueLock(m_QueueLock);

				if (toExec != nullptr)
				{
					toExec = nullptr;
					--m_ToComplete;
				}
				waitForThreads = m_ToComplete != 0;
				if (waitForThreads)
				{
					if (!m_Queue.empty())
					{
						toExec = m_Queue.front();
						m_Queue.pop();
					}
					else
					{
						m_Completed.wait(queueLock);
						waitForThreads = m_ToComplete != 0;
					}
				}
			}
			if (toExec != nullptr)
				toExec();
		}
	}
	else
	{
		std::unique_lock<std::mutex>	queueLock(m_QueueLock);

		while (m_ToComplete != 0)
			m_Completed.wait(queueLock);
	}
	if (m_OnceJobFinished != nullptr)
	{
		m_OnceJobFinished();
		m_OnceJobFinished = nullptr;
	}
}

void	CTaskManager::CallOnceJobFinished(std::function<void()> callback)
{
	m_OnceJobFinished = callback;
}

void	CTaskManager::ConsumerThreadUpdate()
{
	std::function<void()>	toExecute = nullptr;

	while (!m_StopThreads)
	{
		if (toExecute != nullptr)
			toExecute();

		std::unique_lock<std::mutex>	queueLock(m_QueueLock);

		if (toExecute != nullptr)
		{
			toExecute = nullptr;
			--m_ToComplete;
			if (m_ToComplete == 0)
				m_Completed.notify_all();
		}
		if (!m_Queue.empty())
		{
			toExecute = m_Queue.front();
			m_Queue.pop();
			assert(toExecute != nullptr);
		}
		else
		{
			m_QueueChanged.wait(queueLock);
			if (!m_Queue.empty())
			{
				toExecute = m_Queue.front();
				m_Queue.pop();
				assert(toExecute != nullptr);
			}
		}
	}
}
