// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_layer_name.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_layer_name::internalExecute(LensContext *c)
{
}


C_layer_name::C_layer_name(std::string *name, SyntaxError * error)
   : C_production(error), _name(name)
{
}


C_layer_name::C_layer_name(const C_layer_name& rv)
   : C_production(rv), _name(0)
{
   if (rv._name) {
      _name = new std::string(*(rv._name));
   }
}

C_layer_name* C_layer_name::duplicate() const
{
   return new C_layer_name(*this);
}

C_layer_name::~C_layer_name()
{
   delete _name;
}

void C_layer_name::checkChildren() 
{
} 

void C_layer_name::recursivePrint() 
{
   printErrorMessage();
} 
