// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_psetToShared.h"
#include "C_psetMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_psetToShared::execute(MdlContext* context) 
{
   C_psetMapping::execute(context);
}

void C_psetToShared::addToList(C_generalList* gl) 
{  
   std::unique_ptr<C_psetToShared> im;
   im.reset(new C_psetToShared(*this));
   gl->addPSetToShared(std::move(im));
}

C_psetToShared::C_psetToShared() 
   : C_psetMapping()
{
}

C_psetToShared::C_psetToShared(const std::string& psetMember,  
			       C_identifierList* member)
   : C_psetMapping(psetMember, member)
{
} 

void C_psetToShared::duplicate(
   std::unique_ptr<C_psetToShared>&& rv) const
{
   rv.reset(new C_psetToShared(*this));
}

void C_psetToShared::duplicate(
   std::unique_ptr<C_psetMapping>&& rv) const
{
   rv.reset(new C_psetToShared(*this));
}

void C_psetToShared::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_psetToShared(*this));
}

C_psetToShared::~C_psetToShared() 
{
}


