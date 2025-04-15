// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_instanceMapping.h"
#include "C_interfaceMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_instanceMapping::execute(MdlContext* context) 
{
   C_interfaceMapping::execute(context);
}

void C_instanceMapping::addToList(C_generalList* gl) 
{  
   std::unique_ptr<C_instanceMapping> im;
   im.reset(new C_instanceMapping(*this));
   gl->addInstanceMapping(std::move(im));
}


C_instanceMapping::C_instanceMapping() 
   : C_interfaceMapping()
{
}

C_instanceMapping::C_instanceMapping(const std::string& interface,
				     const std::string& interfaceMember,
				     C_identifierList* dataType,
				     bool amp)
   : C_interfaceMapping(interface, interfaceMember, dataType, amp) 
{
} 

void C_instanceMapping::duplicate(std::unique_ptr<C_instanceMapping>&& rv) const
{
   rv.reset(new C_instanceMapping(*this));
}

void C_instanceMapping::duplicate(std::unique_ptr<C_interfaceMapping>&& rv) const
{
   rv.reset(new C_instanceMapping(*this));
}

void C_instanceMapping::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_instanceMapping(*this));
}

C_instanceMapping::~C_instanceMapping() 
{
}


