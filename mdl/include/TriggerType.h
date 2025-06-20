// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TriggerType_H
#define TriggerType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "GslType.h"

class TriggerType : public GslType {
   public:
      virtual void duplicate(std::unique_ptr<DataType>&& rv) const;
      virtual ~TriggerType();        

      virtual std::string getDescriptor() const;
};

#endif // TriggerType_H
