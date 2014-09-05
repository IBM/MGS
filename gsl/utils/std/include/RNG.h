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

#ifndef RNG_H
#define RNG_H

#include "MRG32k3a.h"
//#include "MersenneTwister.h"

typedef MRG32k3a RNG;
//typedef MersenneTwister RNG;

static RNG Rangen;
static RNG SharedRangen;
static RNG& getRangen() {
      return Rangen;
};
static RNG& getSharedRangen() {
      return SharedRangen;
}
;
#endif
