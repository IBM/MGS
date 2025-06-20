// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_name.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_name::internalExecute(GslContext *c)
{
}


C_name::C_name(const C_name& rv)
   : C_production(rv), _name(0)
{
   if (rv._name) {
      _name = new std::string(*(rv._name));
   }
}


C_name::C_name(std::string *name, SyntaxError * error)
   : C_production(error), _name(name)
{
}


C_name* C_name::duplicate() const
{
   return new C_name(*this);
}


C_name::~C_name()
{
   delete _name;
}

void C_name::checkChildren() 
{
} 

void C_name::recursivePrint() 
{
   printErrorMessage();
} 
