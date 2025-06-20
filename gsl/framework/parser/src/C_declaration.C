// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration.h"
#include "C_production.h"

void C_declaration::internalExecute(GslContext *c)
{
}


C_declaration::C_declaration(SyntaxError* error)
   : C_production(error)
{
}

C_declaration::C_declaration(const C_declaration& rv)
   : C_production(rv)
{
}

C_declaration::~C_declaration()
{
}

void C_declaration::checkChildren() 
{
} 

void C_declaration::recursivePrint() 
{
   printErrorMessage();
} 
