// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_functor_category.h"
#include "SyntaxError.h"
#include "C_production.h"
#include <iostream>

void C_functor_category::internalExecute(LensContext *c)
{
}


C_functor_category::C_functor_category(const C_functor_category& rv)
   : C_production(rv), _category(rv._category)
{
}


C_functor_category::C_functor_category(
   std::string s, SyntaxError * error)
   : C_production(error), _category(s)
{
}


C_functor_category* C_functor_category::duplicate() const
{
   return new C_functor_category(*this);
}


C_functor_category::~C_functor_category()
{
}

void C_functor_category::checkChildren() 
{
} 

void C_functor_category::recursivePrint() 
{
   printErrorMessage();
} 
