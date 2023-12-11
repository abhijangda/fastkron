#include <thread>
#include <queue>
#include <mutex>
#include <functional>
#include <condition_variable>

#pragma once

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
  volatile bool* done;

  uint num_threads;
  std::vector<std::mutex> wait_mutexes;
  std::vector<std::mutex> tasks_mutexes;
  std::vector<std::condition_variable> waiting_vars;

  struct thread_args {
    uint thread_id;
    thread_pool* pool;
  };

  static void thread_func(thread_args args) {
    volatile thread_pool* volatile_pool = ((volatile thread_pool*)args.pool);
    while(volatile_pool->is_running()) {
      args.pool->wait_for_task(args.thread_id);
      if (volatile_pool->is_running())
        args.pool->run_task(args.thread_id);
      volatile_pool->thread_done(args.thread_id);
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
    done = new bool[num_threads];
    wait_mutexes = std::vector<std::mutex>(num_threads);
    tasks_mutexes = std::vector<std::mutex>(num_threads);
    waiting_vars = std::vector<std::condition_variable>(num_threads);
    for (uint t = 0; t < num_threads; t++) {
      threads.push_back(std::thread(thread_pool::thread_func, thread_args{t, this}));
      done[t] = false;
    }
  }

  bool is_running() const volatile {return running;}

  void end() {
    running = false;
    notify_all();
    for (uint t = 0; t < num_threads; t++) {
      threads[t].join();
    }
  }

  void run_task(uint id) {
    std::unique_lock<std::mutex> tlk(tasks_mutexes[id]);
    volatile task* t = &tasks[id];
    t->f(t->args);
    tlk.unlock();
  }

  void wait_for_task(int id) {
    std::unique_lock<std::mutex> lk(wait_mutexes[id]);
    waiting_vars[id].wait(lk);
    lk.unlock();
  }

  void notify_all() {
    for (uint i = 0; i < num_threads; i++) {
      done[i] = false;
      std::unique_lock<std::mutex> lk(wait_mutexes[i]);
      waiting_vars[i].notify_all();
      lk.unlock();
    }
  }

  void execute_tasks(task* tasks_) {
    for (uint i = 0; i < num_threads; i++) {
      std::unique_lock<std::mutex> tlk(tasks_mutexes[i]);
      tasks[i] = tasks_[i];
      tlk.unlock();
    }
    notify_all();
  }

  void thread_done(uint thread_id) volatile {
    done[thread_id] = true;
  }

  void join_tasks() {
    for (uint t = 0; t < num_threads; t++) {
      volatile bool* d = (volatile bool*) &done[t];
      while (*d != true);
    }
  }

  ~thread_pool() {
    end();
    join_tasks();
    delete tasks;
  }
};