// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_definition.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_definition::internalExecute(GslContext *c)
{
}


C_definition::C_definition(SyntaxError* error)
   : C_production(error)
{
}

C_definition::C_definition(const C_definition& rv)
   : C_production(rv)
{
}


C_definition* C_definition::duplicate() const
{
   return new C_definition(*this);
}


C_definition::~C_definition()
{
}

void C_definition::checkChildren() 
{
} 

void C_definition::recursivePrint() 
{
   printErrorMessage();
} 
