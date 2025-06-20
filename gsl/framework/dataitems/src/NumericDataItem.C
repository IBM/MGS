// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "NumericDataItem.h"
#include "NDPairTypeChecker.h"

#include <iostream>
#include <sstream>

NumericDataItem& NumericDataItem::operator=(const NumericDataItem& DI)
{
   assign(DI);
   return(*this);
}

#ifdef LINUX

#include "NDPairTypeCheckerNumericCommon.h"

#endif
