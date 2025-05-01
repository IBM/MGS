// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef IzhikUnit_H
#define IzhikUnit_H

#include "Mgs.h"
#include "CG_IzhikUnit.h"
#include "rndm.h"
#include <fstream>
#include "NumIntNoPhase.h"

class IzhikUnit : public CG_IzhikUnit, public RK4NoPhase
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void updateOutputs(RNG& rng);
      virtual void setIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IzhikUnitInAttrPSet* CG_inAttrPset, CG_IzhikUnitOutAttrPSet* CG_outAttrPset);

      virtual ~IzhikUnit();
 protected:
      void derivs(const ShallowArray< double > &, ShallowArray< double > &);
      void outputWeights(std::ofstream& fs);
      //void outputDrivInp(std::ofstream& fs);

};

#endif
