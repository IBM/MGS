// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_argument.h"
#include "C_production.h"

void C_argument::internalExecute(LensContext *c)
{

}

C_argument::C_argument(C_argument::Type t, SyntaxError* error)
   : C_production(error), _type(t)
{
}

C_argument::C_argument(const C_argument& rv)
   : C_production(rv), _type(rv._type)
{
}

C_argument::~C_argument()
{
}

void C_argument::checkChildren() 
{
} 

void C_argument::recursivePrint() 
{
   printErrorMessage();
} 
