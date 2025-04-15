// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LongType_H
#define LongType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "SignedType.h"

class LongType : public SignedType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~LongType();        

      virtual std::string getDescriptor() const;
      virtual std::string getCapitalDescriptor() const;
};
#endif // LongType_H
