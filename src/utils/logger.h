#include <iostream>

#include "env/env.h"

class Logger {
private:
  LogLevel level;

public:
  Logger(LogLevel level) : level(level) {
    if (level <= env::getLogLevel())
      std::cout << "[FastKron] ";
  }

  template <typename T>
  Logger& operator<< (const T &x) {
    if (level <= env::getLogLevel()) {
      std::cout << x;
    }

    return *this;
  }

  Logger& operator<< (std::ostream& (*f)(std::ostream &)) {
    if (level <= env::getLogLevel())
      f(std::cout);
    return *this;
  }

  Logger& operator<< (std::ostream& (*f)(std::ios &)) {
    if (level <= env::getLogLevel())
      f(std::cout);
    return *this;
  }

  Logger& operator<< (std::ostream& (*f)(std::ios_base &)) {
    if (level <= env::getLogLevel())
      f(std::cout);
    return *this;
  }
};