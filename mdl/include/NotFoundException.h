// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NotFoundException_H
#define NotFoundException_H
#include "Mdl.h"

#include "GeneralException.h"

class NotFoundException : public GeneralException {
   public:
      NotFoundException(const std::string& error);
      ~NotFoundException();        
};

#endif // NotFoundException_H
