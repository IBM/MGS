// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_argument_declarator.h"
#include "GslContext.h"
#include "C_declarator.h"
#include "DataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"


void C_argument_declarator::internalExecute(GslContext *c)
{
   _declarator->execute(c);

   _dataitem = const_cast<DataItem*>( 
      c->symTable.getEntry(_declarator->getName()));
   if (!_dataitem) {
      std::string mes = "Argument " + _declarator->getName() + 
	 " was not declared";
      throwError(mes);
   }
}

C_argument_declarator::C_argument_declarator(const C_argument_declarator& rv)
   : C_argument(rv), _dataitem(rv._dataitem), _declarator(0)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
}

C_argument_declarator::C_argument_declarator(
   C_declarator *d, SyntaxError * error)
   : C_argument(_DECLARATOR, error), _dataitem(0), _declarator(d)
{
}

C_argument_declarator* C_argument_declarator::duplicate() const
{
   return new C_argument_declarator(*this);
}

C_argument_declarator::~C_argument_declarator()
{
   delete _declarator;
}

void C_argument_declarator::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
} 

void C_argument_declarator::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   printErrorMessage();
} 
