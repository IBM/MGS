// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_argument_void.h"
#include "DataItem.h"
#include "SyntaxError.h"

void C_argument_void::internalExecute(GslContext *c)
{
}


C_argument_void::C_argument_void(const C_argument_void& rv)
   : C_argument(rv)
{
}


C_argument_void::C_argument_void(SyntaxError * error)
   : C_argument(_NULL, error)
{
}


C_argument_void* C_argument_void::duplicate() const
{
   return new C_argument_void(*this);
}


C_argument_void::~C_argument_void()
{
}


DataItem* C_argument_void::getArgumentDataItem() const
{
   return 0;
}

void C_argument_void::checkChildren() 
{
} 

void C_argument_void::recursivePrint() 
{
   printErrorMessage();
} 
