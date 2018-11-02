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

#ifndef CG_LifeNodeWorkUnitInstance_H
#define CG_LifeNodeWorkUnitInstance_H

#include "Lens.h"
#include "NodePartitionItem.h"
#include "WorkUnit.h"
#include "rndm.h"

class CG_LifeNodeCompCategory;

class CG_LifeNodeWorkUnitInstance : public WorkUnit
{
   public:
      CG_LifeNodeWorkUnitInstance(NodePartitionItem* arg, void (CG_LifeNodeCompCategory::*computeState) (NodePartitionItem*, RNG&), CG_LifeNodeCompCategory* compCategory);
      virtual void execute();
      virtual ~CG_LifeNodeWorkUnitInstance();
   private:
      NodePartitionItem* _arg;
      CG_LifeNodeCompCategory* _compCategory;
      void (CG_LifeNodeCompCategory::*_computeState) (NodePartitionItem*, RNG&);
      RNG _rng;
};

#endif
