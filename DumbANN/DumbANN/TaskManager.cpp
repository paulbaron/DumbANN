#include "TaskManager.h"
#include "DumbANNConfig.h"

#include <algorithm>
#include <assert.h>

#define BATCH_TASK_COUNT				32
#define MIN_VALUES_COMPUTED_PER_TASK	4096

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

void	CTaskManager::MultithreadRange(std::function<void(size_t, size_t)> function, size_t domainSize, size_t threadingHint, bool sync)
{
	CreateThreadsIFN(false);
	size_t		taskCount = std::min(threadingHint / MIN_VALUES_COMPUTED_PER_TASK, m_Threads.size() * 8);

	if (sync)
		WaitForCompletion(true);
	if (taskCount <= 1)
	{
		MICROPROFILE_SCOPEI("CTaskManager", "ExecuteInline", MP_YELLOW4);
		function(0, domainSize); // Not worth multi-threading
	}
	else
	{
		if (taskCount > domainSize)
			taskCount = domainSize;
		std::function<void()>	functions[BATCH_TASK_COUNT] = { };
		size_t					taskIdx = 0;
		size_t					taskIdxInBatch = 0;

		while (taskIdx < taskCount)
		{
			size_t					taskIdxInBatch = 0;

			while (taskIdx < taskCount && taskIdxInBatch < BATCH_TASK_COUNT)
			{
				size_t	range = domainSize / taskCount;
				size_t	minRange = taskIdx * range;
				size_t	maxRange = ((taskIdx + 1) < taskCount) ? minRange + range : domainSize;

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

void	CTaskManager::CreateThreadsIFN(bool forceCreate, int count)
{
	if (m_Threads.empty() || forceCreate)
	{
		size_t	processorCount = (size_t)count;
		if (count < 0)
		{
			processorCount = std::thread::hardware_concurrency();
			processorCount = processorCount <= 1 ? 1 : processorCount - 1;
		}
		if (m_Threads.size() == processorCount)
			return;
		DestroyThreadsIFN();
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
	WaitForCompletion(true);
	m_StopThreads = true;
	m_QueueChanged.notify_all();
	for (int i = 0; i < m_Threads.size(); ++i)
	{
		m_Threads[i]->join();
		delete m_Threads[i];
	}
	m_Threads.clear();
	m_StopThreads = false;
}

void	CTaskManager::WaitForCompletion(bool processTasks)
{
	MICROPROFILE_SCOPEI("CTaskManager", "WaitForCompletion", MP_YELLOW2);
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
#if		ENABLE_MICROPROFILE
	MicroProfileOnThreadCreate("Consumer Thread");
#endif
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
			MICROPROFILE_SCOPEI("CTaskManager", "Pop task", MP_YELLOW);
			toExecute = m_Queue.front();
			m_Queue.pop();
			assert(toExecute != nullptr);
		}
		else
		{
			m_QueueChanged.wait(queueLock);
			if (!m_Queue.empty())
			{
				MICROPROFILE_SCOPEI("CTaskManager", "Pop task", MP_YELLOW);
				toExecute = m_Queue.front();
				m_Queue.pop();
				assert(toExecute != nullptr);
			}
		}
	}
}
