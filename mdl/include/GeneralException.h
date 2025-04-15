// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef GeneralException_H
#define GeneralException_H
#include "Mdl.h"

#include <string>

class GeneralException {
   public:
      GeneralException(const std::string& error);
      const std::string& getError() const;
      void setError(const std::string& error);
      ~GeneralException();        

   private:
      std::string _error;
};

#endif // GeneralException_H
