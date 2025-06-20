// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_general.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>

void C_general::execute(MdlContext* context) 
{
   throw InternalException("C_general::execute is called.");
}

void C_general::addToList(C_generalList* gl) 
{
   throw InternalException("C_general::addToList is called.");
}

C_general::C_general() 
   : C_production()
{

}

C_general::C_general(const C_general& rv) 
   : C_production(rv)
{

}

void C_general::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_general(*this));
}

C_general::~C_general() 
{
}


