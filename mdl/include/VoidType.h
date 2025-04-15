// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VoidType_H
#define VoidType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "DataType.h"

class VoidType : public DataType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~VoidType();        

      virtual std::string getDescriptor() const;
};

#endif // VoidType_h


