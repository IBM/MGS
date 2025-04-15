// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef Operation_H
#define Operation_H
#include "Mdl.h"

#include <memory>
#include <string>

class Predicate;

class Operation {

   public:
      enum _Type {_String, _Bool, _General, _Invalid};
      virtual void duplicate(std::unique_ptr<Operation>&& rv) const;
      virtual ~Operation();
      virtual void operate(Predicate* p1, Predicate* p2, Predicate* cur);

   protected:
      _Type classify(std::string s);

};


#endif // Operation_H
