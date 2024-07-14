#include <iostream>

#include "env/env.h"

class DebugPrint {
private:
  LogLevel level;

public:
  DebugPrint(LogLevel level) : level(level) {
    if (level <= env::getLogLevel())
      std::cout << "[FastKron] ";
  }

  template <typename T>
  DebugPrint& operator<< (const T &x) {
    if (level <= env::getLogLevel()) {
      std::cout << x;
    }

    return *this;
  }

  DebugPrint& operator<< (std::ostream& (*f)(std::ostream &)) {
    if (level <= env::getLogLevel())
      f(std::cout);
    return *this;
  }

  DebugPrint& operator<< (std::ostream& (*f)(std::ios &)) {
    if (level <= env::getLogLevel())
      f(std::cout);
    return *this;
  }

  DebugPrint& operator<< (std::ostream& (*f)(std::ios_base &)) {
    if (level <= env::getLogLevel())
      f(std::cout);
    return *this;
  }
};