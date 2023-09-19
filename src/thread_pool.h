#include <thread>
#include <queue>
#include <mutex>
#include <functional>
#include <condition_variable>

#ifndef __THREAD_POOL__
#define __THREAD_POOL__

template<typename task_args>
class thread_pool {
public:
  struct Task {
    std::function<void(task_args)> f;
    task_args args;
  };

private:
  std::thread* threads;
  Task* tasks;
  const uint num_threads;
  std::mutex mutex;
  std::condition_variable waiting_var;
  
  struct thread_args {
    uint thread_id;
    thread_pool* pool;
    std::mutex& mutex;
  };

  static void thread_func(thread_args args) {
    std::cout << "Thread " << args.thread_id << " running" << std::endl;
    volatile thread_pool* volatile_pool = ((volatile thread_pool*)args.pool);
    while(volatile_pool->is_running()) {
      std::unique_lock<std::mutex> lk(args.pool->mutex);
      args.pool->wait_for_tasks(lk);
      std::cout << "thread " << args.thread_id << " unlocked" << std::endl;
      lk.unlock();
    }
    std::cout << "Thread " << args.thread_id << " finished" << std::endl;
  }

  bool running;

public:
  thread_pool(uint num_threads_) : num_threads(num_threads_), waiting_var() {
    threads = new std::thread[num_threads];
    tasks = new Task[num_threads];
    running = true;

    for (int t = 0; t < num_threads; t++) {
      threads[t] = std::thread(thread_pool::thread_func, thread_args{t, this, mutex});
    }
  }

  bool is_running() const volatile {return running;}

  void end() {
    running = false;
    for (int t = 0; t < num_threads; t++) {
      threads[t].join();
    }
  }

  void wait_for_tasks(std::unique_lock<std::mutex>& lock) {
    waiting_var.wait(lock);
  }

  void execute_tasks() {
    std::unique_lock<std::mutex> lk(mutex);
    lk.unlock();
    waiting_var.notify_all();
  }

  ~thread_pool() {
  }
};

#endif /*__THREAD_POOL__*/