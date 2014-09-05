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
