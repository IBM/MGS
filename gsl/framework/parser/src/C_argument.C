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

#include "C_argument.h"
#include "C_production.h"

void C_argument::internalExecute(LensContext *c)
{

}

C_argument::C_argument(C_argument::Type t, SyntaxError* error)
   : C_production(error), _type(t)
{
}

C_argument::C_argument(const C_argument& rv)
   : C_production(rv), _type(rv._type)
{
}

C_argument::~C_argument()
{
}

void C_argument::checkChildren() 
{
} 

void C_argument::recursivePrint() 
{
   printErrorMessage();
} 
