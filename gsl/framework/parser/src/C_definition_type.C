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
