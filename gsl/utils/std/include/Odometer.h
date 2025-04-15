// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
