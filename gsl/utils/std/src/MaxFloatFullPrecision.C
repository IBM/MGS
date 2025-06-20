// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include <float.h>
#include <math.h>
#include "MaxFloatFullPrecision.h"

MaxFloatFullPrecision maxFloatFullPrecision;

MaxFloatFullPrecision::MaxFloatFullPrecision()
{
   maxvalue = 1;
   for (int i=0;i<FLT_MANT_DIG;i++) maxvalue *= FLT_RADIX;
   maxvalue--;
}
