// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_definition_type.h"
#include "C_type_definition.h"
#include "SyntaxError.h"

void C_definition_type::internalExecute(LensContext *c)
{
   _typeDefinition->execute(c);
}

C_definition_type* C_definition_type::duplicate() const
{
   return new C_definition_type(*this);
}

C_definition_type::C_definition_type(C_type_definition *d, SyntaxError * error)
   : C_definition(error), _typeDefinition(d)
{
}

C_definition_type::C_definition_type(const C_definition_type& rv)
   : C_definition(rv), _typeDefinition(0)
{
   if (rv._typeDefinition) {
      _typeDefinition = rv._typeDefinition->duplicate();
   }
}

C_definition_type::~C_definition_type()
{
   delete _typeDefinition;
}

void C_definition_type::checkChildren() 
{
   if (_typeDefinition) {
      _typeDefinition->checkChildren();
      if (_typeDefinition->isError()) {
         setError();
      }
   }
} 

void C_definition_type::recursivePrint() 
{
   if (_typeDefinition) {
      _typeDefinition->recursivePrint();
   }
   printErrorMessage();
} 
