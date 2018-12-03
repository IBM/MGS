// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
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
      CG_LifeNodeWorkUnitInstance(NodePartitionItem* arg, void (CG_LifeNodeCompCategory::*computeState) (NodePartitionItem*, CG_LifeNodeWorkUnitInstance*), CG_LifeNodeCompCategory* compCategory);
      virtual void execute();
      RNG& getRNG();
      void setGPUMachineID(int GPUMachineID);
      int getGPUMachineID();
      virtual ~CG_LifeNodeWorkUnitInstance();
   private:
      NodePartitionItem* _arg;
      CG_LifeNodeCompCategory* _compCategory;
      void (CG_LifeNodeCompCategory::*_computeState) (NodePartitionItem*, CG_LifeNodeWorkUnitInstance*);
      RNG _rng;
      int _GPUMachineID;
};

#endif
