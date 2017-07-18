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

#ifndef LinskerInfomaxUnit_H
#define LinskerInfomaxUnit_H

#include "Lens.h"
#include "CG_LinskerInfomaxUnit.h"
#include "rndm.h"
#include <fstream>

class LinskerInfomaxUnit : public CG_LinskerInfomaxUnit
{
   public:
      virtual void initialize(RNG& rng);
      virtual void update(RNG& rng);
      virtual void copy(RNG& rng);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LinskerInfomaxUnitInAttrPSet* CG_inAttrPset, CG_LinskerInfomaxUnitOutAttrPSet* CG_outAttrPset);
      virtual void outputWeights(std::ofstream& fsTH, std::ofstream& fsLN);
      virtual void getInputWeights(std::ofstream& fsW);//, std::ofstream& fsdW);
      virtual void getInputWeights(std::vector<double>* W_j);
      virtual void setInputWeights(std::vector<double>* newWeights); 
      virtual ~LinskerInfomaxUnit();
};

#endif
