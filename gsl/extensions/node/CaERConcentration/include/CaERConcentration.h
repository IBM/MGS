#ifndef CaERConcentration_H
#define CaERConcentration_H

#include "Lens.h"
#include "CG_CaERConcentration.h"
#include "rndm.h"

class CaERConcentration : public CG_CaERConcentration
{
   public:
      void initializeCompartmentData(RNG& rng);
      void solve(RNG& rng);
      void finish(RNG& rng);
      virtual void setReceptorCaCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual void setInjectedCaCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual void setProximalJunction(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual bool checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual bool confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual ~CaERConcentration();
};

#endif
