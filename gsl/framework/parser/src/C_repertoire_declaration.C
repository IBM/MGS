// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_repertoire_declaration.h"
#include "C_declarator.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_repertoire_declaration::internalExecute(GslContext *c)
{
   _typeName->execute(c);
   _instanceName->execute(c);
}

std::string const C_repertoire_declaration::getType()
{
   if (_typeName) {
      return _typeName->getName();
   } else {
      _type = "__Empty__";
      return _type;
   }
}

std::string const C_repertoire_declaration::getName()
{
   if (_instanceName) {
      return _instanceName->getName();
   } else {
      _type = "__Empty__";
      return _type;
   }
}

C_repertoire_declaration::C_repertoire_declaration(
   const C_repertoire_declaration& rv)
   : C_production(rv), _typeName(0), _instanceName(0)
{
   if (rv._typeName) {
      _typeName = rv._typeName->duplicate();
   }
   if (rv._instanceName) {
      _instanceName = rv._instanceName->duplicate();
   }
}

C_repertoire_declaration::C_repertoire_declaration(
   C_declarator *type, C_declarator *name, SyntaxError * error)
   : C_production(error), _typeName(type), _instanceName(name)
{
}

C_repertoire_declaration* C_repertoire_declaration::duplicate() const
{
   return new C_repertoire_declaration(*this);
}

C_repertoire_declaration::~C_repertoire_declaration()
{
   delete _typeName;
   delete _instanceName;
}

void C_repertoire_declaration::checkChildren() 
{
   if (_typeName) {
      _typeName->checkChildren();
      if (_typeName->isError()) {
         setError();
      }
   }
   if (_instanceName) {
      _instanceName->checkChildren();
      if (_instanceName->isError()) {
         setError();
      }
   }
} 

void C_repertoire_declaration::recursivePrint() 
{
   if (_typeName) {
      _typeName->recursivePrint();
   }
   if (_instanceName) {
      _instanceName->recursivePrint();
   }
   printErrorMessage();
} 
