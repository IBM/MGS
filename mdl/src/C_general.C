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

#include "C_general.h"
#include "MdlContext.h"
#include "InternalException.h"
#include <memory>

void C_general::execute(MdlContext* context) 
{
   throw InternalException("C_general::execute is called.");
}

void C_general::addToList(C_generalList* gl) 
{
   throw InternalException("C_general::addToList is called.");
}

C_general::C_general() 
   : C_production()
{

}

C_general::C_general(const C_general& rv) 
   : C_production(rv)
{

}

void C_general::duplicate(std::auto_ptr<C_general>& rv) const
{
   rv.reset(new C_general(*this));
}

C_general::~C_general() 
{
}


