// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_dim_declaration.h"
#include "C_int_constant_list.h"
#include "C_production.h"

void C_dim_declaration::internalExecute(LensContext *c)
{
   _intConstantList->execute(c);
}

C_dim_declaration::C_dim_declaration(const C_dim_declaration& rv)
   : C_production(rv), _intConstantList(0)
{
   if (rv._intConstantList) {
      _intConstantList = rv._intConstantList->duplicate();
   }
}

C_dim_declaration::C_dim_declaration(C_int_constant_list *i, 
				     SyntaxError * error)
   : C_production(error), _intConstantList(i)
{
}

C_dim_declaration* C_dim_declaration::duplicate() const
{
   return new C_dim_declaration(*this);
}

const C_int_constant_list* C_dim_declaration::getIntConstantList() const
{
   return _intConstantList;
}

C_dim_declaration::~C_dim_declaration()
{
   delete _intConstantList;
}

void C_dim_declaration::checkChildren() 
{
   if (_intConstantList) {
      _intConstantList->checkChildren();
      if (_intConstantList->isError()) {
         setError();
      }
   }
} 

void C_dim_declaration::recursivePrint() 
{
   if (_intConstantList) {
      _intConstantList->recursivePrint();
   }
   printErrorMessage();
} 
