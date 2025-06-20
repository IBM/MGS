// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SharedPredicate_H
#define SharedPredicate_H
#include "Mdl.h"

#include "Predicate.h"
#include <memory>
#include <string>

class StructType;
class Operation;

class SharedPredicate : public Predicate{

   public:
      SharedPredicate();
      SharedPredicate(Operation* op, const std::string& name);
      virtual void duplicate(std::unique_ptr<Predicate>&& rv) const;
      virtual ~SharedPredicate();
      virtual void setShareds(const MemberContainer<DataType>& shareds);
      virtual bool checkShareds();
      
   private:
      std::string getTypeUsingShared(const MemberContainer<DataType>& shareds);
};


#endif // SharedPredicate_H
