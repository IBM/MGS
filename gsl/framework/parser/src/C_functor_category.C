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
