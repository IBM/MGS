#ifndef IzhikUnit_H
#define IzhikUnit_H

#include "Lens.h"
#include "CG_IzhikUnit.h"
#include "rndm.h"
#include <fstream>
#include "NumIntNoPhase.h"

class IzhikUnit : public CG_IzhikUnit, public RK4
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void updateOutputs(RNG& rng);
      virtual void setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IzhikUnitInAttrPSet* CG_inAttrPset, CG_IzhikUnitOutAttrPSet* CG_outAttrPset);

      virtual ~IzhikUnit();
 protected:
      void derivs(const ShallowArray< double > &, ShallowArray< double > &);
      void outputWeights(std::ofstream& fs);
      //void outputDrivInp(std::ofstream& fs);

};

#endif
