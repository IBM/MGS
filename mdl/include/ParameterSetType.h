// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ParameterSetType_H
#define ParameterSetType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "LensType.h"

class ParameterSetType : public LensType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~ParameterSetType();        

      virtual std::string getDescriptor() const;

      // This method returns if the pointer of the specific dataType 
      // is meant to be owned by the class.
      virtual bool shouldBeOwned() const;
};

#endif // ParameterSetType_H
