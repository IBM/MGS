// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_constructor_clause.h"
#include "C_parameter_type_list.h"
#include "SyntaxError.h"

std::list<C_parameter_type>* C_constructor_clause::getParameterTypeList()
{
   return  _ptl->getList();
}


void C_constructor_clause::internalExecute(GslContext *c)
{
   _ptl->execute(c);
}


C_constructor_clause::C_constructor_clause(const C_constructor_clause& rv)
   : C_complex_functor_clause(rv), _ptl(0)
{
   if (rv._ptl) {
      _ptl = rv._ptl->duplicate();
   }
}


C_constructor_clause::C_constructor_clause(
   C_parameter_type_list *p, SyntaxError * error)
   : C_complex_functor_clause(C_complex_functor_clause::_CONSTRUCTOR, error), 
     _ptl(p)
{
   _ptl = p;
}


C_constructor_clause* C_constructor_clause::duplicate() const
{
   return new C_constructor_clause(*this);
}


C_constructor_clause::~C_constructor_clause()
{
   delete _ptl;
}

void C_constructor_clause::checkChildren() 
{
   if (_ptl) {
      _ptl->checkChildren();
      if (_ptl->isError()) {
         setError();
      }
   }
} 

void C_constructor_clause::recursivePrint() 
{
   if (_ptl) {
      _ptl->recursivePrint();
   }
   printErrorMessage();
} 
