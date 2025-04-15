// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_sharedMapping.h"
#include "C_interfaceMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_sharedMapping::execute(MdlContext* context) 
{
   C_interfaceMapping::execute(context);
}

void C_sharedMapping::addToList(C_generalList* gl) 
{  
   std::unique_ptr<C_sharedMapping> sm;
   sm.reset(new C_sharedMapping(*this));
   gl->addSharedMapping(std::move(sm));
}


C_sharedMapping::C_sharedMapping() 
   : C_interfaceMapping()
{
}

C_sharedMapping::C_sharedMapping(const std::string& interface,
				 const std::string& interfaceMember,
				 C_identifierList* dataType,
				 bool amp)
   : C_interfaceMapping(interface, interfaceMember, dataType, amp)
{
} 

void C_sharedMapping::duplicate(std::unique_ptr<C_sharedMapping>&& rv) const
{
   rv.reset(new C_sharedMapping(*this));
}

void C_sharedMapping::duplicate(std::unique_ptr<C_interfaceMapping>&& rv) const
{
   rv.reset(new C_sharedMapping(*this));
}

void C_sharedMapping::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_sharedMapping(*this));
}

C_sharedMapping::~C_sharedMapping() 
{
}


