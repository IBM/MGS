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

#include "Constant.h"
#include "NDPairList.h"

void Constant::initialize(LensContext *c, const std::vector<DataItem*>& args)
{
   doInitialize(c, args);
}

void Constant::initialize(const NDPairList& ndplist)
{
   doInitialize(ndplist);
}

Constant::~Constant()
{
}

