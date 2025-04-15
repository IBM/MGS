// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef InternalException_H
#define InternalException_H
#include "Mdl.h"

#include "GeneralException.h"

class InternalException : public GeneralException {
   public:
      InternalException(const std::string& error);
      ~InternalException();        
};

#endif // InternalException_H
