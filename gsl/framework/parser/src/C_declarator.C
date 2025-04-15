// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declarator.h"
#include "SyntaxError.h"
#include "C_production.h"

C_declarator::C_declarator(const C_declarator& rv)
   : C_production(rv), _name(0)
{
   if (rv._name) {
      _name = new std::string(*(rv._name));
   } else {
      _name = new std::string("");
   }
}


C_declarator::C_declarator(std::string *name, SyntaxError * error)
   : C_production(error), _name(name)
{
}


C_declarator* C_declarator::duplicate() const
{
   return new C_declarator(*this);
}


void C_declarator::internalExecute(LensContext *c)
{
}


C_declarator::~C_declarator()
{
   delete _name;
}


std::string const& C_declarator::getName() 
{
      // If for some reason it is 0
   if (_name == 0) _name = new std::string("");
   return *_name;
}

void C_declarator::checkChildren() 
{
} 

void C_declarator::recursivePrint() 
{
   printErrorMessage();
} 
