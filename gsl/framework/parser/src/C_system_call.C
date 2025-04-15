// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
