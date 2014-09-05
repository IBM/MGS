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
