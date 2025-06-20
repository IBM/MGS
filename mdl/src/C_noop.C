// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_noop.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include <memory>
#include <string>

void C_noop::execute(MdlContext* context) 
{
}

void C_noop::addToList(C_generalList* gl) 
{
}


C_noop::C_noop() : C_general() 
{

}

C_noop::C_noop(const C_noop& rv) 
   : C_general(rv) 
{
}

void C_noop::duplicate(std::unique_ptr<C_noop>&& rv) const
{
   rv.reset(new C_noop(*this));
}

void C_noop::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_noop(*this));
}

C_noop::~C_noop() 
{
}


