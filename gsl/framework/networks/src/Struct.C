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

#include "Struct.h"
//#include "NDPairList.h"

Struct::Struct()
{
}

void Struct::initialize(LensContext *c, const std::vector<DataItem*>& args)
{
   doInitialize(c, args);
}

void Struct::initialize(const NDPairList& ndplist)
{
   doInitialize(ndplist);
}

Struct::~Struct()
{
}

