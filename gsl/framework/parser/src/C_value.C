// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_value.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_value::internalExecute(LensContext *c)
{

}

C_value::C_value(const C_value& rv)
   : C_production(rv), _value(0)
{
   if (rv._value) {
      _value = new std::string(*(rv._value));
   }
}

C_value::C_value(std::string *value, SyntaxError * error)
   : C_production(error), _value(value)
{
}

C_value* C_value::duplicate() const
{
   return new C_value(*this);
}

C_value::~C_value()
{
   delete _value;
}

void C_value::checkChildren() 
{
} 

void C_value::recursivePrint() 
{
   printErrorMessage();
} 
