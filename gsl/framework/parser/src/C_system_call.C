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

#include "C_system_call.h"
#include "SyntaxError.h"
#include "C_production.h"
#include <stdio.h>
#include <stdlib.h>

C_system_call::C_system_call(const C_system_call& rv)
   : C_production(rv), _command(0)
{
   if (rv._command) {
      _command = new std::string(*(rv._command));
   }
}

C_system_call::C_system_call(std::string *command, SyntaxError * error)
   : C_production(error), _command(command)
{
}

C_system_call* C_system_call::duplicate() const
{
   return new C_system_call(*this);
}

void C_system_call::internalExecute(LensContext *c)
{
   int s=system(_command->c_str());
}

C_system_call::~C_system_call()
{
   delete _command;
}

void C_system_call::checkChildren() 
{
} 

void C_system_call::recursivePrint() 
{
   printErrorMessage();
} 
