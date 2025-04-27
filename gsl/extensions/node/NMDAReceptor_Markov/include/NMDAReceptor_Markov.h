#ifndef NMDAReceptor_Markov_H
#define NMDAReceptor_Markov_H

#include "Lens.h"
#include "CG_NMDAReceptor_Markov.h"
#include "rndm.h"

#include "MaxComputeOrder.h"
#include <iostream>
#include <fstream>
#include <sstream>

#define DEBUG_CHAN 


#if RECEPTOR_NMDA == NMDAR_POINTPROCESS
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#elif RECEPTOR_NMDA == NMDAR_BEHABADI_2012 || \
      RECEPTOR_NMDA == NMDAR_BEHABADI_2012_MODIFIED
#define BASED_TEMPERATURE 35.0  // Celcius
#define Q10 3.0
#else
#define BASED_TEMPERATURE 0.0  // Celcius
#define Q10 3.0
#endif

#ifndef Q10 
#define Q10 3.0 //default
#endif
class NMDAReceptor_Markov : public CG_NMDAReceptor_Markov
{
   public:
      void initializeNMDA(RNG& rng);
      void updateNMDA(RNG& rng);
      void updateNMDADepPlasticity(RNG& rng);
      virtual void setPostIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptor_MarkovInAttrPSet* CG_inAttrPset, CG_NMDAReceptor_MarkovOutAttrPSet* CG_outAttrPset);
      virtual void setPrePostIndex(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptor_MarkovInAttrPSet* CG_inAttrPset, CG_NMDAReceptor_MarkovOutAttrPSet* CG_outAttrPset);
      virtual ~NMDAReceptor_Markov();
  #ifdef DEBUG_CHAN
      NMDAReceptor_Markov();
  #endif
  #ifdef MICRODOMAIN_CALCIUM
  virtual void setCalciumMicrodomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NMDAReceptor_MarkovInAttrPSet* CG_inAttrPset, CG_NMDAReceptor_MarkovOutAttrPSet* CG_outAttrPset);
  int _offset; //the offset due to the presence of different Ca2+-microdomain
  #endif

  private:
   #ifdef DEBUG_CHAN
   std::ofstream* outFile;
   #endif
};

#endif
