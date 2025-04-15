// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _MAXFLOATFULLPRECISION_
#define _MAXFLOATFULLPRECISION_
#include "Copyright.h"

class MaxFloatFullPrecision
{
   public:
      float value(){return maxvalue;}
      MaxFloatFullPrecision();

   private:
      float maxvalue;
};

extern MaxFloatFullPrecision maxFloatFullPrecision;
#endif
