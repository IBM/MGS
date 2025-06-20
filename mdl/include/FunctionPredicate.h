// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FunctionPredicate_H
#define FunctionPredicate_H
#include "Mdl.h"

#include "Predicate.h"
#include <memory>
#include <string>
#include <vector>

class StructType;
class Operation;
class PredicateFunction;

class FunctionPredicate : public Predicate
{
   public:
      FunctionPredicate();
      FunctionPredicate(Operation* op, const std::string& name);
      virtual void duplicate(std::unique_ptr<Predicate>&& rv) const;
      virtual ~FunctionPredicate();
      virtual void setFunctionPredicateName(
	 std::vector<PredicateFunction*>* functions);
      virtual void getFunctionPredicateNames(
	 std::set<std::string>& names) const;
   private:
      std::string _originalName;
};


#endif // FunctionPredicate_H
