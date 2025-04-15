// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_complex_functor_clause.h"
#include "C_parameter_type_list.h"

#include "C_constructor_clause.h"
#include "C_function_clause.h"
#include "C_return_clause.h"
#include "C_production.h"

C_complex_functor_clause::C_complex_functor_clause(Type t, SyntaxError* error)
   : C_production(error), _type(t)
{
}

C_complex_functor_clause::C_complex_functor_clause(
   const C_complex_functor_clause& rv)
   : C_production(rv), _type(rv._type)
{
}


C_complex_functor_clause::~C_complex_functor_clause()
{
}

void C_complex_functor_clause::checkChildren() 
{
} 

void C_complex_functor_clause::recursivePrint() 
{
   printErrorMessage();
} 
