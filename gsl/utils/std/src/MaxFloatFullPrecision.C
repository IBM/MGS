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
