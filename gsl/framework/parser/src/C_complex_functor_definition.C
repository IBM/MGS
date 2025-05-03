// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_complex_functor_definition.h"
#include "C_functor_category.h"
#include "C_declarator.h"
#include "C_complex_functor_declaration_body.h"
#include "SyntaxError.h"
#include "C_production.h"
#include <iostream>

void C_complex_functor_definition::internalExecute(GslContext *c)
{
   _functorCategory->execute(c);
   _declarator->execute(c);
   _complexFunctorDec->execute(c);
}

C_complex_functor_definition::C_complex_functor_definition(
   const C_complex_functor_definition& rv)
   : C_production(rv), _functorCategory(0), _declarator(0), 
     _complexFunctorDec(0)
{
   if (rv._functorCategory) {
      _functorCategory = rv._functorCategory->duplicate();
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
   if (rv._complexFunctorDec) {
      _complexFunctorDec = rv._complexFunctorDec->duplicate();
   }
}

C_complex_functor_definition::C_complex_functor_definition(
   C_functor_category *c, C_declarator *d, 
   C_complex_functor_declaration_body *b, SyntaxError * error)
   : C_production(error), _functorCategory(c), _declarator(d), 
     _complexFunctorDec(b)
{
}

C_complex_functor_definition* C_complex_functor_definition::duplicate() const
{
   return new C_complex_functor_definition(*this);
}

C_complex_functor_definition::~C_complex_functor_definition()
{
   delete _functorCategory;
   delete _declarator;
   delete _complexFunctorDec;
}

std::string const & C_complex_functor_definition::getCategory() const
{
   return _functorCategory->getCategory();
}

std::string const & C_complex_functor_definition::getName() 
{
   if (_declarator) {
      return _declarator->getName();
   } else {
      _name = "__Empty__";
      return _name;
   }
}

std::list<C_parameter_type>* 
C_complex_functor_definition::getConstructorParameters()
{
   return _complexFunctorDec->getConstructorParameters();
}

std::list<C_parameter_type>* 
C_complex_functor_definition::getFunctionParameters()
{
   return _complexFunctorDec->getFunctionParameters();
}

std::list<C_parameter_type>* 
C_complex_functor_definition::getReturnParameters()
{
   return _complexFunctorDec->getReturnParameters();
}

void C_complex_functor_definition::checkChildren() 
{
   if (_functorCategory) {
      _functorCategory->checkChildren();
      if (_functorCategory->isError()) {
         setError();
      }
   }
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
   if (_complexFunctorDec) {
      _complexFunctorDec->checkChildren();
      if (_complexFunctorDec->isError()) {
         setError();
      }
   }
} 

void C_complex_functor_definition::recursivePrint() 
{
   if (_functorCategory) {
      _functorCategory->recursivePrint();
   }
   if (_declarator) {
      _declarator->recursivePrint();
   }
   if (_complexFunctorDec) {
      _complexFunctorDec->recursivePrint();
   }
   printErrorMessage();
} 
