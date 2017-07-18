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
