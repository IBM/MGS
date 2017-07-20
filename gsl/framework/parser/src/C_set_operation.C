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

#include "C_set_operation.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_set_operation::internalExecute(LensContext *c)
{
}


C_set_operation::C_set_operation(const C_set_operation& rv)
   : C_production(rv), _type(rv._type)
{
}


C_set_operation::C_set_operation(std::string s, SyntaxError * error)
   : C_production(error)
{
   if (s=="set") {
      _type = _SET;
   }
   if (s=="copy") {
      _type = _COPY;
   }
   _error = error;
}


C_set_operation* C_set_operation::duplicate() const
{
   return new C_set_operation(*this);
}


C_set_operation::~C_set_operation()
{
}

void C_set_operation::checkChildren() 
{
} 

void C_set_operation::recursivePrint() 
{
   printErrorMessage();
} 
