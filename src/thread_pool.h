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
  struct task {
    void(*f)(task_args);
    task_args args;

    task(void(*f_)(task_args), task_args args_) : 
      f(f_), args(args_) {}
    
    task() {}

    volatile task& operator= (const task& x) volatile {
      f = x.f;
      args = x.args;
      return *this;
    }

    volatile task& operator= (const volatile task& x) volatile {
      f = x.f;
      args = x.args;
      return *this;
    }
  };

private:
  std::vector<std::thread> threads;
  volatile task* tasks;
  std::vector<char> done;

  uint num_threads;
  std::mutex wait_mutex;
  std::condition_variable waiting_var;

  struct thread_args {
    uint thread_id;
    thread_pool* pool;
  };

  static void thread_func(thread_args args) {
    volatile thread_pool* volatile_pool = ((volatile thread_pool*)args.pool);
    while(volatile_pool->is_running()) {
      std::unique_lock<std::mutex> lk(args.pool->wait_mutex);
      args.pool->wait_for_tasks(lk);
      lk.unlock();
      
      volatile_pool->run_task(args.thread_id);
      args.pool->thread_done(args.thread_id);
    }
  }

  bool running;

public:
  thread_pool(): num_threads(0) {}

  thread_pool(uint num_threads_) : num_threads(num_threads_) {
    init(num_threads_);
  }

  void init(uint num_threads_) {
    num_threads = num_threads_;
    running = true;
    tasks = new task[num_threads];

    for (uint t = 0; t < num_threads; t++) {
      threads.push_back(std::thread(thread_pool::thread_func, thread_args{t, this}));
      done.push_back(false);
    }
  }

  bool is_running() const volatile {return running;}

  void end() {
    running = false;
    for (uint t = 0; t < num_threads; t++) {
      threads[t].join();
    }
  }

  void run_task (uint thread_id) volatile {
    volatile task* t = &tasks[thread_id];
    t->f(t->args);
  }

  void wait_for_tasks(std::unique_lock<std::mutex>& lock) {
    waiting_var.wait(lock);
  }

  void execute_tasks(task* tasks_) {
    std::unique_lock<std::mutex> lk(wait_mutex);
    for (uint i = 0; i < num_threads; i++) {
      tasks[i] = tasks_[i];
    }
    waiting_var.notify_all();
    lk.unlock();
  }

  void thread_done(uint thread_id) {
    done[thread_id] = true;
  }

  void join_tasks() {
    for (uint t = 0; t < num_threads; t++) {
      volatile bool* d = (volatile bool*) &done[t];
      while (*d != true);
      *d = false;
    }
  }

  ~thread_pool() {
    //TODO: join
  }
};

#endif /*__THREAD_POOL__*/
