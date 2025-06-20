// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PredicateException_H
#define PredicateException_H
#include "Mdl.h"

#include "GeneralException.h"

class PredicateException : public GeneralException {
   public:
      PredicateException(const std::string& error);
      ~PredicateException();        
};

#endif // PredicateException_H
