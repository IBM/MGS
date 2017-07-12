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
