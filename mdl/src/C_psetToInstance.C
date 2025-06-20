// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_psetToInstance.h"
#include "C_psetMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_psetToInstance::execute(MdlContext* context) 
{
   C_psetMapping::execute(context);
}

void C_psetToInstance::addToList(C_generalList* gl) 
{  
   std::unique_ptr<C_psetToInstance> im;
   im.reset(new C_psetToInstance(*this));
   gl->addPSetToInstance(std::move(im));
}

C_psetToInstance::C_psetToInstance() 
   : C_psetMapping()
{
}

C_psetToInstance::C_psetToInstance(const std::string& psetMember,
				   C_identifierList* member)
   : C_psetMapping(psetMember, member)
{
} 

void C_psetToInstance::duplicate(
   std::unique_ptr<C_psetToInstance>&& rv) const
{
   rv.reset(new C_psetToInstance(*this));
}

void C_psetToInstance::duplicate(
   std::unique_ptr<C_psetMapping>&& rv) const
{
   rv.reset(new C_psetToInstance(*this));
}

void C_psetToInstance::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_psetToInstance(*this));
}

C_psetToInstance::~C_psetToInstance() 
{
}


