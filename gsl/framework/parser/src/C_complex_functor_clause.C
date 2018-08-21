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
