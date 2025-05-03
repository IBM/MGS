// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Struct.h"
//#include "NDPairList.h"

Struct::Struct()
{
}

void Struct::initialize(GslContext *c, const std::vector<DataItem*>& args)
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

