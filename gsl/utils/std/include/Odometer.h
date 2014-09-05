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
