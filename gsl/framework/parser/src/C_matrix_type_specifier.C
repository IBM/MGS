// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_matrix_type_specifier.h"
#include "C_type_specifier.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_matrix_type_specifier::internalExecute(GslContext *c)
{
   _typeSpecifier->execute(c);
}


C_matrix_type_specifier::C_matrix_type_specifier(
   const C_matrix_type_specifier& rv)
   : C_production(rv), _typeSpecifier(0)
{
   if (rv._typeSpecifier) {
      _typeSpecifier = rv._typeSpecifier->duplicate();
   }
}


C_matrix_type_specifier::C_matrix_type_specifier(
   C_type_specifier *ts, SyntaxError * error)
   : C_production(error), _typeSpecifier(ts)
{
}


C_matrix_type_specifier* C_matrix_type_specifier::duplicate() const
{
   return new C_matrix_type_specifier(*this);
}


C_matrix_type_specifier::~C_matrix_type_specifier()
{
   delete _typeSpecifier;
}


C_type_specifier * C_matrix_type_specifier::getTypeSpecifier() const
{
   return _typeSpecifier;
}

void C_matrix_type_specifier::checkChildren() 
{
   if (_typeSpecifier) {
      _typeSpecifier->checkChildren();
      if (_typeSpecifier->isError()) {
         setError();
      }
   }
} 

void C_matrix_type_specifier::recursivePrint() 
{
   if (_typeSpecifier) {
      _typeSpecifier->recursivePrint();
   }
   printErrorMessage();
} 
