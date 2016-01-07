#ifndef CaERConcentration_H
#define CaERConcentration_H

#include "Lens.h"
#include "CG_CaERConcentration.h"
#include "rndm.h"

class CaERConcentration : public CG_CaERConcentration
{
   public:
      void initializeCaConcentration(RNG& rng);
      void solve(RNG& rng);
      void finish(RNG& rng);
      virtual void setReceptorCaCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual void setInjectedCaCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual void setProximalJunction(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual bool checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual bool confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset);
      virtual ~CaERConcentration();

	  //added
	  // - specific to one compartment variable
      float getLambda(DimensionStruct* a, DimensionStruct* b);
      float getArea(DimensionStruct* a, DimensionStruct* b);
      float getArea(int i);
      float getVolume(DimensionStruct* a, DimensionStruct* b);
      float getVolume(int i);
      float getAij(DimensionStruct* a, DimensionStruct* b, float Volume);
      unsigned getSize() {return branchData->size;}
	
	  // - common for all compartment variables
      void initializeCompartments(RNG& rng);
#if MAX_COMPUTE_ORDER>0
      void forwardSolve1(RNG& rng);
      void backwardSolve1(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>1
      void forwardSolve2(RNG& rng);
      void backwardSolve2(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>2
      void forwardSolve3(RNG& rng);
      void backwardSolve3(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>3
      void forwardSolve4(RNG& rng);
      void backwardSolve4(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>4
      void forwardSolve5(RNG& rng);
      void backwardSolve5(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>5
      void forwardSolve6(RNG& rng);
      void backwardSolve6(RNG& rng);
#endif
#if MAX_COMPUTE_ORDER>6
      void forwardSolve7(RNG& rng);
      void backwardSolve7(RNG& rng);
#endif
   private:
      void doForwardSolve();
      void doBackwardSolve();
};

#endif
