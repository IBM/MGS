// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

  std::string _gslFile;
  bool _enableErd;
  int _numWorkUnits;
  bool _readGraph;
  bool _outputGraph;
  bool _simulate;
  bool _verbose;
};
#endif
