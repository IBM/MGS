/* =================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-07-18-2017

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

=================================================================
*/


#ifndef ChannelLeak_H
#define ChannelLeak_H

#include "Lens.h"
#include "CG_ChannelLeak.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "SegmentDescriptor.h"
#include <fstream> 


class ChannelLeak : public CG_ChannelLeak
{
   public:
      void update(RNG& rng);
      void initialize(RNG& rng);
      virtual ~ChannelLeak();
};

#endif
