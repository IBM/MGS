// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CG_LifeNodeWorkUnitShared_H
#define CG_LifeNodeWorkUnitShared_H

#include "Lens.h"
#include "WorkUnit.h"
#include "rndm.h"

class CG_LifeNodeCompCategory;
class LifeNodeCompCategory;

class CG_LifeNodeWorkUnitShared : public WorkUnit
{
   public:
      CG_LifeNodeWorkUnitShared(void (LifeNodeCompCategory::*computeState) (RNG&), CG_LifeNodeCompCategory* compCategory);
      virtual void execute();
      virtual ~CG_LifeNodeWorkUnitShared();
   private:
      LifeNodeCompCategory* _compCategory;
      void (LifeNodeCompCategory::*_computeState) (RNG&);
      RNG _rng;
};

#endif
