// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef MahonUnit_H
#define MahonUnit_H

#include "Mgs.h"
#include "CG_MahonUnit.h"
#include "rndm.h"
#include <fstream>
#include "NumInt.h"

class MahonUnit : public CG_MahonUnit, public RK4Phased
{
   public:
      using Node::initialize;
      void initialize(RNG& rng);
      void update1(RNG& rng);
      void update2(RNG& rng);
      void update3(RNG& rng);
      void update4(RNG& rng);
      void flushVars1(RNG& rng);
      void flushVars2(RNG& rng);
      void flushVars3(RNG& rng);
      void flushVars4(RNG& rng);
      void updateOutputs(RNG& rng);
      virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MahonUnitInAttrPSet* CG_inAttrPset, CG_MahonUnitOutAttrPSet* CG_outAttrPset);
      virtual bool ConnectP1(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MahonUnitInAttrPSet* CG_inAttrPset, CG_MahonUnitOutAttrPSet* CG_outAttrPset);

      virtual ~MahonUnit();
 protected:
       void derivs(const ShallowArray< double > &, ShallowArray< double > &);
       void outputWeights(std::ofstream& fs);
       //void outputDrivInp(std::ofstream& fs);
       void flushVars(const ShallowArray< double > &);
      //void flushVars_x1();
};

#endif
