// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef EdgeTypeType_H
#define EdgeTypeType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "GslType.h"

class EdgeTypeType : public GslType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~EdgeTypeType();        

      virtual std::string getDescriptor() const;
};

#endif // EdgeTypeType_H
