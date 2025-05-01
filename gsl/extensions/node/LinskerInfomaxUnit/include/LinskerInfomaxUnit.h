// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LinskerInfomaxUnit_H
#define LinskerInfomaxUnit_H

#include "Mgs.h"
#include "CG_LinskerInfomaxUnit.h"
#include "rndm.h"
#include <fstream>

class LinskerInfomaxUnit : public CG_LinskerInfomaxUnit
{
   public:
      virtual void initialize(RNG& rng);
      virtual void update(RNG& rng);
      virtual void copy(RNG& rng);
      virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LinskerInfomaxUnitInAttrPSet* CG_inAttrPset, CG_LinskerInfomaxUnitOutAttrPSet* CG_outAttrPset);
      virtual void outputWeights(std::ofstream& fsTH, std::ofstream& fsLN);
      virtual void getInputWeights(std::ofstream& fsW);//, std::ofstream& fsdW);
      virtual void getInputWeights(std::vector<double>* W_j);
      virtual void setInputWeights(std::vector<double>* newWeights); 
      virtual ~LinskerInfomaxUnit();
};

#endif
