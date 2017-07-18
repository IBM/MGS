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

#include "C_init_attr_type_edge.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_init_attr_type_edge::internalExecute(LensContext *c)
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
