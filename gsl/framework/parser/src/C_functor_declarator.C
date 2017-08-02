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

#include "C_functor_declarator.h"
#include "SyntaxError.h"
#include "C_production.h"
#include <iostream>

void C_functor_declarator::internalExecute(LensContext *c)
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
