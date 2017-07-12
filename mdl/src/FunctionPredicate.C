// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Predicate.h"
#include "FunctionPredicate.h"
#include "PredicateFunction.h"
#include "StructType.h"
#include "Operation.h"
#include "InternalException.h"
#include "Constants.h"
#include "ConnectionException.h"
#include <memory>
#include <string>
#include <iostream>

FunctionPredicate::FunctionPredicate() 
   : Predicate(), _originalName("")
{
}

FunctionPredicate::FunctionPredicate(Operation* op, const std::string& name)
   : Predicate(op, name), _originalName(name)
{
}

void FunctionPredicate::duplicate(std::auto_ptr<Predicate>& rv) const
{
   rv.reset(new FunctionPredicate(*this));
}

void FunctionPredicate::setFunctionPredicateName(
   std::vector<PredicateFunction*>* functions) 
{
   Predicate::setFunctionPredicateName(functions);
   _type = "bool";
   if(_name == "") {
      throw InternalException(
 	 "_name is empty in FunctionPredicate::getTypeUsingFunction");
   }
   bool error = true;
   if (functions) {
      std::vector<PredicateFunction*>::const_iterator it, 
	 end = functions->end();
      for (it = functions->begin(); it != end; ++it) {
	 if ((*it)->getName() == _name) {
	    error = false;
	    break;
	 }
      }
   } 
   if (error) {
      throw ConnectionException(
	 _name + " is not defined as a PredicateFunction.");
   }
   _name = PREDICATEFUNCTIONPREFIX + _name;
}

FunctionPredicate::~FunctionPredicate() 
{
}


void FunctionPredicate::getFunctionPredicateNames(
   std::set<std::string>& names) const
{
   names.insert(_originalName);
}
