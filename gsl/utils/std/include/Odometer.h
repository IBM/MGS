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

#ifndef ODOMETER_H
#define ODOMETER_H
#include "Copyright.h"

#include <vector>


class Odometer
{

   public:

      virtual bool isAtEnd() =0;
      virtual bool isRolledOver() =0;
      virtual std::vector<int> & look() = 0;
      virtual std::vector<int> & next() =0;
      virtual void reset() =0;
      virtual int getSize() =0;
      virtual ~Odometer() {}
};

#endif
