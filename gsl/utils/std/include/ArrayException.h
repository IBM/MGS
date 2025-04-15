// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ArrayException_h
#define ArrayException_h
#include "Copyright.h"

#include <string>

class ArrayException {
 public:
  ArrayException(const std::string& error) : _error(error) {}
    const std::string& getError() {return _error;}
    void setError(const std::string& error) {_error=error;}
    ~ArrayException() {}
 private:
    std::string _error;
};

#endif // __ArrayException_h
