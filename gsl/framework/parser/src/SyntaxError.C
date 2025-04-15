// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "SyntaxError.h"

#include <iostream>
#include <sstream>
#include <string>

// Turn on or off the full trace printing
const bool FULLTRACE = false;

SyntaxError::SyntaxError() : _errorMessage(""), _error(false) {}

// TUAN: TODO: plan to convert from char* to std::string
SyntaxError::SyntaxError(std::string& fileName, int lineNum, const char* prod,
                         const char* rule, const char* errMsg, bool err)
    : _error(err), _original(err) {
  std::ostringstream os;
  os << "File " << fileName << ":" << lineNum << ": While parsing " << prod;
  std::string rule_string(rule);
  if (not rule_string.empty()) {
    // strcmp(rule, "") != 0) {
    os << ", using " << rule;
  }
  std::string errMsg_string(errMsg);
  if (not errMsg_string.empty()) {
    //    if (strcmp(errMsg, "") != 0) {
    os << ", " << errMsg;
  }
  _errorMessage = os.str();
}

SyntaxError::SyntaxError(SyntaxError* c)
    : _errorMessage(c->_errorMessage),
      _error(c->_error),
      _original(c->_original) {}

SyntaxError* SyntaxError::duplicate() { return new SyntaxError(this); }

SyntaxError::~SyntaxError() {}

bool SyntaxError::isError() { return _error; }

void SyntaxError::setError(bool error) { _error = error; }

void SyntaxError::printMessage() {
  if ((FULLTRACE || _original) && _error) {
    std::cerr << _errorMessage << std::endl;
  }
}

void SyntaxError::appendMessage(const std::string& err) {
  _errorMessage += ", ";
  _errorMessage += err;
}
