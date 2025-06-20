// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef COMMANDLINE_H
#define COMMANDLINE_H
#include "Copyright.h"

#include <vector>
#include <string>


class CommandLine
{
 public:
  CommandLine(bool verbose);
  bool parse(int argc, char** argv);
  ~CommandLine();

  int getGpuID() const {
    return _gpuID;
  }

#ifndef DISABLE_PTHREADS
  int getThreads() const {
    return _threads;
  }
  const std::string& getUserInterface() const {
    return _userInterface;
  }
  int getGuiPort() const {
    return _guiPort;
  }
  bool getBindCpus() const {
    return _bindCpus;
  }
#endif // DISABLE_PTHREADS

  unsigned getSeed() const {
    return _seed;
  }

  const std::string& getGslFile() const {
    return _gslFile;
  }
  bool getEnableErd() const {
    return _enableErd;
  }

  int getWorkUnits() {
    return _numWorkUnits;
  }

  bool getReadGraph() {
    return _readGraph;
  }

  bool getOutputGraph() {
    return _outputGraph;
  }

  bool getSimulate() const {
    return _simulate;
  }

 private:
  bool convertToInt(const std::string& str, int& num);
      
#ifndef DISABLE_PTHREADS
  int _threads;
  std::string _userInterface;
  int _guiPort;
  bool _bindCpus;
#endif // DISABLE_PTHREADS
  unsigned _seed;
  std::string _gslFile;
  bool _enableErd;
  int _numWorkUnits;
  bool _readGraph;
  bool _outputGraph;
  bool _simulate;
  bool _verbose;
  int _gpuID;
};
#endif
