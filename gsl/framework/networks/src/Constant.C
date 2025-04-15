// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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

