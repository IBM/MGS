#ifndef BengioRatePyramidal_H
#define BengioRatePyramidal_H

#include "Lens.h"
#include "CG_BengioRatePyramidal.h"
#include "rndm.h"

class BengioRatePyramidal : public CG_BengioRatePyramidal
{
   public:
      void update_U(RNG& rng);
      void update_Vs(RNG& rng);
      virtual void setLateralIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BengioRatePyramidalInAttrPSet* CG_inAttrPset, CG_BengioRatePyramidalOutAttrPSet* CG_outAttrPset);
      virtual ~BengioRatePyramidal();
};

#endif
