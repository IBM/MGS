// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef InstancePredicate_H
#define InstancePredicate_H
#include "Mdl.h"

#include "Predicate.h"
#include <memory>
#include <string>

class StructType;
class Operation;

class InstancePredicate : public Predicate{

   public:
      InstancePredicate();
      InstancePredicate(Operation* op, const std::string& name);
      virtual void duplicate(std::unique_ptr<Predicate>&& rv) const;
      virtual ~InstancePredicate();
      virtual void setInstances(const MemberContainer<DataType>& instances);
      
   private:
      std::string getTypeUsingInstance(const MemberContainer<DataType>& instances);
};


#endif // InstancePredicate_H
