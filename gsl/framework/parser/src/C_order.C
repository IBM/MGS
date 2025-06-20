// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_order.h"
#include "C_int_constant_list.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_order::internalExecute(GslContext *c)
{
   _cIntConstList->execute(c);
}

C_order::C_order(const C_order& rv)
   : C_production(rv), _cIntConstList(0)
{
   if (rv._cIntConstList) {
      _cIntConstList = rv._cIntConstList->duplicate();
   }
}

C_order::C_order(C_int_constant_list *c, SyntaxError * error)
   : C_production(error), _cIntConstList(c)
{
}

const std::list<int>* C_order::getListInt() const
{
   return _cIntConstList->getList();
}

C_order* C_order::duplicate() const
{
   return new C_order(*this);
}

C_order::~C_order()
{
   delete _cIntConstList;
}

void C_order::checkChildren() 
{
   if (_cIntConstList) {
      _cIntConstList->checkChildren();
      if (_cIntConstList->isError()) {
         setError();
      }
   }
} 

void C_order::recursivePrint() 
{
   if (_cIntConstList) {
      _cIntConstList->recursivePrint();
   }
   printErrorMessage();
} 
