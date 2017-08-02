// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
