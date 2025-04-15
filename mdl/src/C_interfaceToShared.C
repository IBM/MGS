// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_interfaceToShared.h"
#include "C_interfaceMapping.h"
#include "C_generalList.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>
#include <string>

void C_interfaceToShared::execute(MdlContext* context) 
{
   C_interfaceMapping::execute(context);
}

void C_interfaceToShared::addToList(C_generalList* gl) 
{  
   std::unique_ptr<C_interfaceToShared> im;
   im.reset(new C_interfaceToShared(*this));
   gl->addInterfaceToShared(std::move(im));
}

C_interfaceToShared::C_interfaceToShared() 
   : C_interfaceMapping()
{
}

C_interfaceToShared::C_interfaceToShared(
   const std::string& interface, const std::string& interfaceMember,
   C_identifierList* member)
   : C_interfaceMapping(interface, interfaceMember, member)
{
} 

void C_interfaceToShared::duplicate(
   std::unique_ptr<C_interfaceToShared>&& rv) const
{
   rv.reset(new C_interfaceToShared(*this));
}

void C_interfaceToShared::duplicate(
   std::unique_ptr<C_interfaceMapping>&& rv) const
{
   rv.reset(new C_interfaceToShared(*this));
}

void C_interfaceToShared::duplicate(std::unique_ptr<C_general>&& rv) const
{
   rv.reset(new C_interfaceToShared(*this));
}

C_interfaceToShared::~C_interfaceToShared() 
{
}


