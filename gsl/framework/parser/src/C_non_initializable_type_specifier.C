// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_non_initializable_type_specifier.h"
#include "C_type_specifier.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_non_initializable_type_specifier::internalExecute(GslContext *c)
{
}


C_non_initializable_type_specifier::C_non_initializable_type_specifier(
   const C_non_initializable_type_specifier& rv)
   : C_production(rv), _type(rv._type), _nextTypeSpec(0)
{
}


C_non_initializable_type_specifier::C_non_initializable_type_specifier(
   C_type_specifier::Type t, SyntaxError * error)
   : C_production(error), _type(t), _nextTypeSpec(0)
{
}


C_non_initializable_type_specifier* 
C_non_initializable_type_specifier::duplicate() const
{
   return new C_non_initializable_type_specifier(*this);
}


C_non_initializable_type_specifier::~C_non_initializable_type_specifier()
{
}

void C_non_initializable_type_specifier::checkChildren() 
{
   if (_nextTypeSpec) {
      _nextTypeSpec->checkChildren();
      if (_nextTypeSpec->isError()) {
         setError();
      }
   }
} 

void C_non_initializable_type_specifier::recursivePrint() 
{
   if (_nextTypeSpec) {
      _nextTypeSpec->recursivePrint();
   }
   printErrorMessage();
} 
