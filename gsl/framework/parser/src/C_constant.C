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

#include "C_constant.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_constant::internalExecute(LensContext *c)
{
}


C_constant::C_constant(int i, SyntaxError * error)
   : C_production(error), _type(_INT), _intValue(i), _floatValue(0)
{
}


C_constant *C_constant::duplicate() const
{
   return new C_constant(*this);
}


C_constant::C_constant(double f, SyntaxError * error)
   : C_production(error), _type(_FLOAT), _intValue(0), _floatValue(f)
{
}


C_constant::C_constant(const C_constant& rv)
   : C_production(rv), _type(rv._type), _intValue(rv._intValue), 
     _floatValue(rv. _floatValue)
{
}


float C_constant::getFloat()
{
   float retVal;
   if (_type==_INT) retVal = float(_intValue);
   else retVal = _floatValue;
   return retVal;
}


int C_constant::getInt()
{
   int retVal;
   if (_type==_INT) retVal = _intValue;
   else retVal = int(_floatValue);
   return retVal;
}


C_constant::~C_constant()
{
}

void C_constant::checkChildren() 
{
} 

void C_constant::recursivePrint() 
{
   printErrorMessage();
} 
