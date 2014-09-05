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

#include "Variable.h"

void Variable::initialize(LensContext *c, const std::vector<DataItem*>& args)
{
   doInitialize(c, args);
}

void Variable::initialize(const NDPairList& ndplist)
{
   doInitialize(ndplist);
}

Variable::~Variable()
{
}

