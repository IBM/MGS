// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_functor_declarator.h"
#include "SyntaxError.h"
#include "C_production.h"
#include <iostream>

void C_functor_declarator::internalExecute(GslContext *c)
{
}


C_functor_declarator::C_functor_declarator(const C_functor_declarator& rv)
   : C_production(rv), _id(rv._id)
{
}


C_functor_declarator::C_functor_declarator(
   const std::string& s, SyntaxError * error)
   : C_production(error), _id(s)
{
}


C_functor_declarator* C_functor_declarator::duplicate() const
{
   return new C_functor_declarator(*this);
}


C_functor_declarator::~C_functor_declarator()
{
}

void C_functor_declarator::checkChildren() 
{
} 

void C_functor_declarator::recursivePrint() 
{
   printErrorMessage();
} 
