// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PSetPredicate_H
#define PSetPredicate_H
#include "Mdl.h"

#include "Predicate.h"
#include <memory>
#include <string>

class StructType;
class Operation;

class PSetPredicate : public Predicate{

   public:
      PSetPredicate();
      PSetPredicate(Operation* op, const std::string& name);
      virtual void duplicate(std::unique_ptr<Predicate>&& rv) const;
      virtual ~PSetPredicate();
      virtual void setPSet(StructType& type);
      
   private:
      std::string getTypeUsingPSet(StructType& type);
};


#endif // PSetPredicate_H
