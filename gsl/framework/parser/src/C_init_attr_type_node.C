// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_init_attr_type_node.h"
#include "ParameterSet.h"
#include "SyntaxError.h"
#include "C_production.h"

void C_init_attr_type_node::internalExecute(LensContext *c)
{

}


C_init_attr_type_node::C_init_attr_type_node(const C_init_attr_type_node& rv)
   : C_production(rv), _type(rv._type)
{
}


C_init_attr_type_node::C_init_attr_type_node(int _t, SyntaxError * error)
   : C_production(error)
{
   switch(_t) {
      case 0:
         _type=_IN;
         break;
      case 1:
         _type=_OUT;
         break;
      case 2:
         _type=_NODEINIT;
         break;
   }
}


C_init_attr_type_node* C_init_attr_type_node::duplicate() const
{
   return new C_init_attr_type_node(*this);
}


C_init_attr_type_node::~C_init_attr_type_node()
{
}


std::string C_init_attr_type_node::getModelType()
{
   std::string retval;
   if (_type == _IN )
      retval = "IN";
   if ( _type == _OUT )
      retval = "OUT";
   if ( _type == _NODEINIT )
      retval = "INIT";
   return retval;
}

void C_init_attr_type_node::checkChildren() 
{
} 

void C_init_attr_type_node::recursivePrint() 
{
   printErrorMessage();
} 
