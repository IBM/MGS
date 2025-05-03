// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_int.h"
#include "GslContext.h"
#include "C_declarator.h"
#include "C_constant.h"
#include "IntDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_int::internalExecute(GslContext *c)
{
   _declarator->execute(c);

   // transfer data to DataItem
   IntDataItem *fdi = new IntDataItem;
   fdi->setInt(_intValue);
   std::unique_ptr<DataItem> fdi_ap(fdi);
   try {
      c->symTable.addEntry(_declarator->getName(), fdi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring int, " + e.getError());
   }
}


C_declaration_int* C_declaration_int::duplicate() const
{
   return new C_declaration_int(*this);
}


C_declaration_int::C_declaration_int(const C_declaration_int& rv)
   : C_declaration(rv), _declarator(0), _intValue(rv._intValue)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
}


C_declaration_int::~C_declaration_int()
{
   delete _declarator;
}


C_declaration_int::C_declaration_int(
   C_declarator *d, int i, SyntaxError * error)
   : C_declaration(error), _declarator(d), _intValue(i)
{
}

void C_declaration_int::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
} 

void C_declaration_int::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   printErrorMessage();
} 
