// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_init_attr_type_edge.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_init_attr_type_edge::internalExecute(GslContext *c)
{

}


C_init_attr_type_edge::C_init_attr_type_edge(const C_init_attr_type_edge& rv)
   : C_production(rv)
{
}


C_init_attr_type_edge::C_init_attr_type_edge(SyntaxError * error)
   : C_production(error)
{
}


C_init_attr_type_edge* C_init_attr_type_edge::duplicate() const
{
   return new C_init_attr_type_edge(*this);
}


C_init_attr_type_edge::~C_init_attr_type_edge()
{
}

void C_init_attr_type_edge::checkChildren() 
{
} 

void C_init_attr_type_edge::recursivePrint() 
{
   printErrorMessage();
} 
