// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_return_clause.h"
#include "C_parameter_type_list.h"
#include "SyntaxError.h"

void C_return_clause::internalExecute(GslContext *c)
{
   _ptl->execute(c);
}

C_return_clause::C_return_clause(const C_return_clause& rv)
   : C_complex_functor_clause(rv), _ptl(0)
{
   if (rv._ptl) {
      _ptl = rv._ptl->duplicate();
   }
}


C_return_clause::C_return_clause(C_parameter_type_list *p, SyntaxError * error)
   : C_complex_functor_clause(C_complex_functor_clause::_RETURN, error), 
     _ptl(p)
{
}


C_return_clause* C_return_clause::duplicate() const
{
   return new C_return_clause(*this);
}


C_return_clause::~C_return_clause()
{
   delete _ptl;
}

std::list<C_parameter_type>* C_return_clause::getParameterTypeList() 
{
   return _ptl->getList();
}

void C_return_clause::checkChildren() 
{
   if (_ptl) {
      _ptl->checkChildren();
      if (_ptl->isError()) {
         setError();
      }
   }
} 

void C_return_clause::recursivePrint() 
{
   if (_ptl) {
      _ptl->recursivePrint();
   }
   printErrorMessage();
} 
