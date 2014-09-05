// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_definition.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_definition::internalExecute(LensContext *c)
{
}


C_definition::C_definition(SyntaxError* error)
   : C_production(error)
{
}

C_definition::C_definition(const C_definition& rv)
   : C_production(rv)
{
}


C_definition* C_definition::duplicate() const
{
   return new C_definition(*this);
}


C_definition::~C_definition()
{
}

void C_definition::checkChildren() 
{
} 

void C_definition::recursivePrint() 
{
   printErrorMessage();
} 
