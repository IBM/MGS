// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
   std::auto_ptr<C_initialize> init;
   init.reset(new C_initialize(*this));
   gl->addInitialize(init);
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

void C_initialize::duplicate(std::auto_ptr<C_initialize>& rv) const
{
   rv.reset(new C_initialize(*this));
}

void C_initialize::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_initialize(*this));
}

C_initialize::~C_initialize() 
{
}
