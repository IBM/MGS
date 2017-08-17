#include "Lens.h"
#include "AMPAReceptor_Markov.h"
#include "CG_AMPAReceptor_Markov.h"
#include "rndm.h"
#include <iostream>
#include "math.h"
#include <limits>
#include "NTSMacros.h"
#include "NumberUtils.h"
#include "StringUtils.h"

#define SMALL 1.0E-6
#define decimal_places 6
#define fieldDelimiter "\t"

// dynamics of neurotransmitter
#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
  #define NEUROTRANSMITTER      \
    (getSharedMembers().NTmax / \
     (1.0 + exp(-(*Vpre - getSharedMembers().Vp) / getSharedMembers().Kp)))
#elif SYNAPSE_MODEL_STRATEGY == USE_SYNAPTICCLEFT 
// NOTE: (should be updated in SynapticCleft nodetype)
#define NEUROTRANSMITTER      *Glut
#endif

// take into account the effect of temperature making the change faster or
// slower
#define DT (*(getSharedMembers().deltaT))
#define Tscale (*(getSharedMembers().deltaT) * (getSharedMembers().Tadj))

// Destexhe-Mainen-Sejnowski (1994)
// (1.3.1)
// The gating of AMPAR is modeled using
//  Patneau - Mayer (1991) Neuron - Kinetic analysis of interaction between
//  Kainate and AMPA
//  NOTE: Here there are 2 desensitized states
//        ...
//                     1         2             3
//   Kainate,AMPA + R <=> R_Kai <=>  R(Kai)_2 <=>[alpha][beta] O
//                         /\          /\
//                       4 ||          ||  5
//                         \/          \/
//                         RD(AMPA)   RD(AMPA)_2
//   1. <=>[Rb* NT][Ru1]
//   2. <=>[Rb* NT][Ru2]
//   3. <=>[alpha][beta]
//   4. <=>[Rd][Rr]
//   5. <=>[Rd][Rr]
// then given r = fO (fraction of AMPAR in Open state)
//    dfO/dt = alpha * fC2 - beta * fO
#define Ro (getSharedMembers().alpha)
#define Rc (getSharedMembers().beta)
#define Rb1 (getSharedMembers().Rb)
#define Rb2 (getSharedMembers().Rb)
//#define Rb2 (getSharedMembers().Rb*NEUROTRANSMITTER) - bad
#define Ru1 (getSharedMembers().Ru1)
#define Ru2 (getSharedMembers().Ru2)
#define Rr (getSharedMembers().Rr)
#define Rd (getSharedMembers().Rd)
#define Rr2 (getSharedMembers().Rr2)
#define Rd2 (getSharedMembers().Rd2)

void AMPAReceptor_Markov::initializeAMPA(RNG& rng)
{
#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
  assert(Vpre);
#endif
  dyn_var_t r = fO;

  // std::cout << "Weight: " << (*w) << std::endl;
  if (w == NULL)
  {
    g = gbar * r;
  }
  else
  {
    g = (*w) * gbar * r;
  }
  // initialize
  fC0 = 1.0;
  fC1 = fD1 = fC2 = fD2 = fO = 0.0;
  assert(fabs(fC0 + fC1 + fD1 + fC2 + fD2 + fO - 1.0) < SMALL);  // conservation
  #ifdef DEBUG_CHAN
    std::ostringstream os;
    std::ostringstream os2;
    std::string fileName = "AMPAR_Markov.dat";
    os << fileName << StringUtils::random_string(3) << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
    (*outFile) << "#Time" << fieldDelimiter << ": States";
    os2 << "# time, C0, C1, C2, D1, D2, O" ;
    (*outFile) << "\n";
    (*outFile) << os2.str() << "\n";
  #endif
}

void AMPAReceptor_Markov::updateAMPA(RNG& rng)
{
  fO = Tscale * (Ro * fC2 - Rc * fO) + fO;
  fC2 = Tscale * (Rb2 * NEUROTRANSMITTER * fC1 + Rc * fO + Rr * fD2 -
              (Ro + Ru2 + Rd) * fC2) +
        fC2;
  fC1 = Tscale * (Rb1 * NEUROTRANSMITTER * fC0 + Ru2 * fC2 + Rr * fD1 -
              (Rb2 * NEUROTRANSMITTER + Ru1 + Rd) * fC1) +
        fC1;
  fD1 = Tscale * (Rd * fC1 - Rr * fD1) + fD1;
  fD2 = Tscale * (Rd * fC2 - Rr * fD2) + fD2;

  if (fO < 0.0) { fO = 0.0; }
  else if (fO > 1.0) { fO = 1.0; }
  if (fC2 < 0.0) { fC2 = 0.0; }
  else if (fC2 > 1.0) { fC2 = 1.0; }
  if (fC1 < 0.0) { fC1 = 0.0; }
  else if (fC1 > 1.0) { fC1 = 1.0; }
  if (fD1 < 0.0) { fD1 = 0.0; }
  else if (fD1 > 1.0) { fD1 = 1.0; }
  if (fD2 < 0.0) { fD2 = 0.0; }
  else if (fD2 > 1.0) { fD2 = 1.0; }
  fC0 = 1.0 - (fC1 + fD1 + fC2 + fD2 + fO);

  if(w==NULL){
    g = gbar*fO;}
  else{
    g = (*w)*gbar*fO;
  }
  I = g * ((*Vpost)[indexPost] - getSharedMembers().E);
  #ifdef DEBUG_CHAN
    (*outFile) << float(getSimulation().getIteration()) * DT;
    (*outFile) << std::fixed << fieldDelimiter << fC0;
    (*outFile) << std::fixed << fieldDelimiter << fC1;
    (*outFile) << std::fixed << fieldDelimiter << fC2;
    (*outFile) << std::fixed << fieldDelimiter << fD1;
    (*outFile) << std::fixed << fieldDelimiter << fD2;
    (*outFile) << std::fixed << fieldDelimiter << fO;
    (*outFile) << "\n";
  #endif
}

void AMPAReceptor_Markov::setPostIndex(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_AMPAReceptor_MarkovInAttrPSet* CG_inAttrPset,
    CG_AMPAReceptor_MarkovOutAttrPSet* CG_outAttrPset)
{
  indexPost = CG_inAttrPset->idx;
  if (indexPrePost.size() % 2)
  {//it means that PreSynapticPoint is being used
#ifdef KEEP_PAIR_PRE_POST
    indexPrePost.push_back(&indexPost);
#endif
  }
}

AMPAReceptor_Markov::~AMPAReceptor_Markov() 
{
  #ifdef DEBUG_CHAN
    if (outFile) 
    {
      outFile->close();
    }
    delete outFile;
  #endif
}
#ifdef DEBUG_CHAN
AMPAReceptor_Markov::AMPAReceptor_Markov() 
   : CG_AMPAReceptor_Markov(), outFile(0)
{
}
#endif

void AMPAReceptor_Markov::setPrePostIndex(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_AMPAReceptor_MarkovInAttrPSet* CG_inAttrPset,
    CG_AMPAReceptor_MarkovOutAttrPSet* CG_outAttrPset)
{
  indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[0]);
  indexPrePost.push_back(&(*(getSharedMembers().indexPrePost_connect))[1]);
}
