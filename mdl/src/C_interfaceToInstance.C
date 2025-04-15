// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_interfaceToInstance.h"
#include "C_interfaceMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_interfaceToInstance::execute(MdlContext* context) 
{
   C_interfaceMapping::execute(context);
}

void C_interfaceToInstance::addToList(C_generalList* gl) 
{  
   std::unique_ptr<C_interfaceToInstance> im;
   im.reset(new C_interfaceToInstance(*this));
   gl->addInterfaceToInstance(std::move(im));
}

C_interfaceToInstance::C_interfaceToInstance() 
   : C_interfaceMapping()
{
}

C_interfaceToInstance::C_interfaceToInstance(
   const std::string& interface, const std::string& interfaceMember,
   C_identifierList* member)
   : C_interfaceMapping(interface, interfaceMember, member)
{
} 

void C_interfaceToInstance::duplicate(
   std::unique_ptr<C_interfaceToInstance>&& rv) const
{
   rv.reset(new C_interfaceToInstance(*this));
}

void C_interfaceToInstance::duplicate(
   std::unique_ptr<C_interfaceMapping>&& rv) const
{
   rv.reset(new C_interfaceToInstance(*this));
}

void C_interfaceToInstance::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_interfaceToInstance(*this));
}

C_interfaceToInstance::~C_interfaceToInstance() 
{
}


