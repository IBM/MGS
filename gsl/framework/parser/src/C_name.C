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

#include "C_name.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_name::internalExecute(LensContext *c)
{
}


C_name::C_name(const C_name& rv)
   : C_production(rv), _name(0)
{
   if (rv._name) {
      _name = new std::string(*(rv._name));
   }
}


C_name::C_name(std::string *name, SyntaxError * error)
   : C_production(error), _name(name)
{
}


C_name* C_name::duplicate() const
{
   return new C_name(*this);
}


C_name::~C_name()
{
   delete _name;
}

void C_name::checkChildren() 
{
} 

void C_name::recursivePrint() 
{
   printErrorMessage();
} 
