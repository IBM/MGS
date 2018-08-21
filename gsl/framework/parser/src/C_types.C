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

#include "C_types.h"
#include "SyntaxError.h"
#include "C_production.h"

C_types::C_types(SyntaxError * error)
   : C_production(error)
{
}

C_types::C_types(const C_types& rv)
   : C_production(rv), _type(rv._type)
{
}

void C_types::internalExecute(LensContext *c)
{
}

C_types* C_types::duplicate() const
{
   return new C_types(*this);
}

C_types::~C_types()
{
}

void C_types::checkChildren() 
{
} 

void C_types::recursivePrint() 
{
   printErrorMessage();
} 
