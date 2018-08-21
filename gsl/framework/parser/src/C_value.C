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
