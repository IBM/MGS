// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
   std::auto_ptr<C_instanceMapping> im;
   im.reset(new C_instanceMapping(*this));
   gl->addInstanceMapping(im);
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

void C_instanceMapping::duplicate(std::auto_ptr<C_instanceMapping>& rv) const
{
   rv.reset(new C_instanceMapping(*this));
}

void C_instanceMapping::duplicate(std::auto_ptr<C_interfaceMapping>& rv) const
{
   rv.reset(new C_instanceMapping(*this));
}

void C_instanceMapping::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_instanceMapping(*this));
}

C_instanceMapping::~C_instanceMapping() 
{
}


