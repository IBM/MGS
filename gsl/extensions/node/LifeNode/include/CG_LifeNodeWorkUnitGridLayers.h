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

#ifndef CG_LifeNodeWorkUnitGridLayers_H
#define CG_LifeNodeWorkUnitGridLayers_H

#include "Lens.h"
#include "GridLayerData.h"
#include "WorkUnit.h"
#include "rndm.h"

class CG_LifeNodeCompCategory;

class CG_LifeNodeWorkUnitGridLayers : public WorkUnit
{
   public:
      CG_LifeNodeWorkUnitGridLayers(GridLayerData* arg, void (CG_LifeNodeCompCategory::*computeState) (GridLayerData*, RNG&), CG_LifeNodeCompCategory* compCategory);
      virtual void execute();
      virtual ~CG_LifeNodeWorkUnitGridLayers();
   private:
      GridLayerData* _arg;
      CG_LifeNodeCompCategory* _compCategory;
      void (CG_LifeNodeCompCategory::*_computeState) (GridLayerData*, RNG&);
      RNG _rng;
};

#endif
