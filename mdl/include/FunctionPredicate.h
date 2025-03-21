// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
