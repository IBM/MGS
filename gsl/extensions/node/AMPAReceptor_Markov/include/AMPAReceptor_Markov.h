#ifndef AMPAReceptor_Markov_H
#define AMPAReceptor_Markov_H

#include "Lens.h"
#include "CG_AMPAReceptor_Markov.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

#define DEBUG_CHAN
#include <iostream>
#include <fstream>
#include <sstream>

#if RECEPTOR_AMPA == AMPAR_Markov_DESTEXHE_MAINEN_SEJNOWSKI_1994
 // The parameter is based upon Xiang, GreenWood, Brown - 1992
 // recorded at 31^oC
#define BASED_TEMPERATURE 31.0  // Celcius
#define Q10 3.0
#endif

#ifndef BASED_TEMPERATURE
#define BASED_TEMPERATURE 35.0  // Celcius
#endif
#ifndef Q10 
#define Q10 3.0 //default
#endif
class AMPAReceptor_Markov : public CG_AMPAReceptor_Markov
{
   public:
      void updateAMPA(RNG& rng);
      void initializeAMPA(RNG& rng);
      virtual void setPostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AMPAReceptor_MarkovInAttrPSet* CG_inAttrPset, CG_AMPAReceptor_MarkovOutAttrPSet* CG_outAttrPset);
      virtual void setPrePostIndex(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AMPAReceptor_MarkovInAttrPSet* CG_inAttrPset, CG_AMPAReceptor_MarkovOutAttrPSet* CG_outAttrPset);
      virtual ~AMPAReceptor_Markov();
  #ifdef DEBUG_CHAN
      AMPAReceptor_Markov();
  #endif
   #ifdef DEBUG_CHAN
   std::ofstream* outFile;
   #endif
};

#endif
