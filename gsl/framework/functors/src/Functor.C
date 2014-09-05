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

#include "Functor.h"
#include <iostream>

const char * Functor::_category = "FUNCTOR";

Functor::Functor()
{
}


const char * Functor::getCategory()
{
   return _category;
}


void Functor::initialize(LensContext *c, const std::vector<DataItem*>& args)
{
   doInitialize(c, args);
}


void Functor::execute(LensContext *c, 
		      const std::vector<DataItem*>& args, 
		      std::auto_ptr<DataItem>& rvalue)
{
   doExecute(c, args, rvalue);
}


Functor::~Functor()
{
}
