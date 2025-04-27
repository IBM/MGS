#ifndef SingleChannelRYR_H
#define SingleChannelRYR_H

#include "Lens.h"
#include "CG_SingleChannelRYR.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#ifndef BASED_TEMPERATURE
#define BASED_TEMPERATURE 35.0  // Celcius
#endif

#ifndef Q10
#define Q10 2.3  // default
#endif
class SingleChannelRYR : public CG_SingleChannelRYR
{
  public:
  void update(RNG& rng);
  void initialize(RNG& rng);
  virtual ~SingleChannelRYR();

  //user-defined
  void updateChannelTransitionRate(dyn_var_t*& matChannelTransitionRate,
                                   int cptIdx);
  #ifdef MICRODOMAIN_CALCIUM
  virtual void setCalciumMicrodomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SingleChannelRYRInAttrPSet* CG_inAttrPset, CG_SingleChannelRYROutAttrPSet* CG_outAttrPset);
  int _offset; //the offset due to the presence of different Ca2+-microdomain
  #endif


  private:
};

#endif
