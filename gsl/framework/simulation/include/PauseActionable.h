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

#ifndef PAUSEACTIONABLE_H
#define PAUSEACTIONABLE_H
#include "Copyright.h"


class PauseActionable
{
   public:
      virtual void action() =0;
      virtual ~PauseActionable() {}
};
#endif
