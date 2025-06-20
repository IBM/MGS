// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_initialize.h"
#include "C_argumentToMemberMapper.h"
#include "C_general.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_initialize::execute(MdlContext* context) 
{
}

void C_initialize::addToList(C_generalList* gl) 
{
   std::unique_ptr<C_initialize> init;
   init.reset(new C_initialize(*this));
   gl->addInitialize(std::move(init));
}

std::string C_initialize::getType() const
{
   return "Initialize";
}

C_initialize::C_initialize(bool ellipsisIncluded) 
   : C_argumentToMemberMapper(ellipsisIncluded)
{

}

C_initialize::C_initialize(C_generalList* argumentList, bool ellipsisIncluded)
   : C_argumentToMemberMapper(argumentList, ellipsisIncluded)
{

}


C_initialize::C_initialize(const C_initialize& rv) 
   : C_argumentToMemberMapper(rv)
{
}

void C_initialize::duplicate(std::unique_ptr<C_initialize>&& rv) const
{
   rv.reset(new C_initialize(*this));
}

void C_initialize::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_initialize(*this));
}

C_initialize::~C_initialize() 
{
}
