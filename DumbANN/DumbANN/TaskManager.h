#pragma once

#include <queue>
#include <functional>
#include <mutex>

class	CTaskManager
{
public:
	CTaskManager();
	~CTaskManager();

	void		MultithreadRange(	std::function<void(size_t,size_t)> function,
									size_t domainSize,
									size_t threadingHint,
									bool sync = true);
	void		DestroyThreadsIFN();
	void		CreateThreadsIFN(bool forceCreate, int count = -1);
	void		WaitForCompletion(bool processTasks = false);
	void		CallOnceJobFinished(std::function<void()> callback);

private:
	void		ConsumerThreadUpdate();

	std::vector<std::thread*>			m_Threads;
	std::queue<std::function<void()> >	m_Queue;
	std::mutex							m_QueueLock;
	std::condition_variable				m_QueueChanged;
	std::condition_variable				m_Completed;
	int									m_ToComplete;
	std::function<void()>				m_OnceJobFinished;
	bool								m_StopThreads;
};