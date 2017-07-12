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

#include "C_layer_name.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_layer_name::internalExecute(LensContext *c)
{
}


C_layer_name::C_layer_name(std::string *name, SyntaxError * error)
   : C_production(error), _name(name)
{
}


C_layer_name::C_layer_name(const C_layer_name& rv)
   : C_production(rv), _name(0)
{
   if (rv._name) {
      _name = new std::string(*(rv._name));
   }
}

C_layer_name* C_layer_name::duplicate() const
{
   return new C_layer_name(*this);
}

C_layer_name::~C_layer_name()
{
   delete _name;
}

void C_layer_name::checkChildren() 
{
} 

void C_layer_name::recursivePrint() 
{
   printErrorMessage();
} 
