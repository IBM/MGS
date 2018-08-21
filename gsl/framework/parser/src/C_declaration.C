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

#include "C_declaration.h"
#include "C_production.h"

void C_declaration::internalExecute(LensContext *c)
{
}


C_declaration::C_declaration(SyntaxError* error)
   : C_production(error)
{
}

C_declaration::C_declaration(const C_declaration& rv)
   : C_production(rv)
{
}

C_declaration::~C_declaration()
{
}

void C_declaration::checkChildren() 
{
} 

void C_declaration::recursivePrint() 
{
   printErrorMessage();
} 
