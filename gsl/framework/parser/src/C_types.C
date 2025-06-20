// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_types.h"
#include "SyntaxError.h"
#include "C_production.h"

C_types::C_types(SyntaxError * error)
   : C_production(error)
{
}

C_types::C_types(const C_types& rv)
   : C_production(rv), _type(rv._type)
{
}

void C_types::internalExecute(GslContext *c)
{
}

C_types* C_types::duplicate() const
{
   return new C_types(*this);
}

C_types::~C_types()
{
}

void C_types::checkChildren() 
{
} 

void C_types::recursivePrint() 
{
   printErrorMessage();
} 
