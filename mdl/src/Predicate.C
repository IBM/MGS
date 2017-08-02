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

#include "Predicate.h"
#include "StructType.h"
#include "Operation.h"
#include "InternalException.h"
#include <memory>
#include <string>
#include <iostream>

Predicate::Predicate() 
   : _type(""), _name(""), _predicate1(0), _predicate2(0), _operation(0)
{
}

Predicate::Predicate(Operation* op, Predicate* p1, Predicate* p2) 
   : _type(""), _name(""), _predicate1(p1), _predicate2(p2), 
     _operation(op)
{
}

Predicate::Predicate(Operation* op, const std::string& name,
		     const std::string& type) 
   : _type(type), _name(name), _predicate1(0), _predicate2(0), 
     _operation(op)
{
}

Predicate::Predicate(const Predicate& rv) 
   : _type(rv._type), _name(rv._name), _predicate1(0), _predicate2(0), 
     _operation(0)
{
   if (rv._predicate1) {
      std::auto_ptr<Predicate> dup;
      rv._predicate1->duplicate(dup);
      _predicate1 = dup.release();
   }
   if (rv._predicate2) {
      std::auto_ptr<Predicate> dup;
      rv._predicate2->duplicate(dup);
      _predicate2 = dup.release();
   }
   if (rv._operation) {
      std::auto_ptr<Operation> dup;
      rv._operation->duplicate(dup);
      _operation = dup.release();
   }
}

void Predicate::duplicate(std::auto_ptr<Predicate>& rv) const
{
   rv.reset(new Predicate(*this));
}

std::string Predicate::getResult() 
{
   operate();
   return _name;
}

void Predicate::setPSet(StructType& type) 
{
   if (_predicate1) {
      _predicate1->setPSet(type);
   }
   if (_predicate2) {
      _predicate2->setPSet(type);
   }
}

void Predicate::setInstances(const MemberContainer<DataType>& instances)
{
   if (_predicate1) {
      _predicate1->setInstances(instances);
   }
   if (_predicate2) {
      _predicate2->setInstances(instances);
   }
}

void Predicate::setShareds(const MemberContainer<DataType>& shareds)
{
   if (_predicate1) {
      _predicate1->setShareds(shareds);
   }
   if (_predicate2) {
      _predicate2->setShareds(shareds);
   }
}

bool Predicate::checkShareds()
{
   bool retval = false;
   if (_predicate1) {
      retval = retval || _predicate1->checkShareds();
   }
   if (_predicate2) {
      retval = retval || _predicate2->checkShareds();
   }
   return retval;
}

void Predicate::setFunctionPredicateName(
   std::vector<PredicateFunction*>* functions)
{
   if (_predicate1) {
      _predicate1->setFunctionPredicateName(functions);
   }
   if (_predicate2) {
      _predicate2->setFunctionPredicateName(functions);
   }
}

std::string Predicate::getType() const
{
   return _type;
}

void Predicate::setType(const std::string& type) 
{
   _type = type;
}

std::string Predicate::getName() const
{
   return _name;
}

void Predicate::setName(const std::string& name) 
{
   _name = name;
}

void Predicate::operate() 
{
   if (_operation == 0) {
      throw InternalException("_operation is 0 in Predicate::getResult");
   }
   _operation->operate(_predicate1, _predicate2, this);
}

Predicate::~Predicate() 
{
   delete _predicate1;
   delete _predicate2;
   delete _operation;
}

void Predicate::getFunctionPredicateNames(
   std::set<std::string>& names) const
{
   if (_predicate1) {
      _predicate1->getFunctionPredicateNames(names);
   }
   if (_predicate2) {
      _predicate2->getFunctionPredicateNames(names);
   }
}
