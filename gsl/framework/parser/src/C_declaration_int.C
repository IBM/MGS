// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_declaration_int.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "C_constant.h"
#include "IntDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"

void C_declaration_int::internalExecute(LensContext *c)
{
   _declarator->execute(c);

   // transfer data to DataItem
   IntDataItem *fdi = new IntDataItem;
   fdi->setInt(_intValue);
   std::auto_ptr<DataItem> fdi_ap(fdi);
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
