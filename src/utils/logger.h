#include <iostream>

#include "env/env.h"

#pragma once

class Logger {
private:
  LogLevel level;

public:
  Logger(LogLevel level) : level(level) {
    if (valid())
      std::cout << "[FastKron] ";
  }

  template <typename T>
  Logger& operator<< (const T &x) {
    if (valid()) {
      std::cout << x;
    }

    return *this;
  }

  Logger& operator<< (std::ostream& (*f)(std::ostream &)) {
    if (valid())
      f(std::cout);
    return *this;
  }

  Logger& operator<< (std::ostream& (*f)(std::ios &)) {
    if (valid())
      f(std::cout);
    return *this;
  }

  Logger& operator<< (std::ostream& (*f)(std::ios_base &)) {
    if (valid())
      f(std::cout);
    return *this;
  }

  bool valid() {return level <= env::getLogLevel();}
};