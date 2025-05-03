// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Variable.h"

void Variable::initialize(GslContext *c, const std::vector<DataItem*>& args)
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

