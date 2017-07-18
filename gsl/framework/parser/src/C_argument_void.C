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

#include "C_argument_void.h"
#include "DataItem.h"
#include "SyntaxError.h"

void C_argument_void::internalExecute(LensContext *c)
{
}


C_argument_void::C_argument_void(const C_argument_void& rv)
   : C_argument(rv)
{
}


C_argument_void::C_argument_void(SyntaxError * error)
   : C_argument(_NULL, error)
{
}


C_argument_void* C_argument_void::duplicate() const
{
   return new C_argument_void(*this);
}


C_argument_void::~C_argument_void()
{
}


DataItem* C_argument_void::getArgumentDataItem() const
{
   return 0;
}

void C_argument_void::checkChildren() 
{
} 

void C_argument_void::recursivePrint() 
{
   printErrorMessage();
} 
