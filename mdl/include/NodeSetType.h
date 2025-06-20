// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NodeSetType_H
#define NodeSetType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "GslType.h"

class NodeSetType : public GslType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~NodeSetType();        

      virtual std::string getDescriptor() const;

      // This method returns if the pointer of the specific dataType 
      // is meant to be owned by the class.
      virtual bool shouldBeOwned() const;
};

#endif // NodeSetType_H
