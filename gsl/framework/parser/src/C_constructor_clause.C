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

#include "C_constructor_clause.h"
#include "C_parameter_type_list.h"
#include "SyntaxError.h"

std::list<C_parameter_type>* C_constructor_clause::getParameterTypeList()
{
   return  _ptl->getList();
}


void C_constructor_clause::internalExecute(LensContext *c)
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
