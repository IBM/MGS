#include "Lens.h"
#include "AMPAReceptor_Markov.h"
#include "CG_AMPAReceptor_Markov.h"
#include "rndm.h"

#define SMALL 1.0E-6
// Destexhe-Mainen-Sejnowski (1994)
// The Glutamate neurotransmitter concentration, i.e. [NT]
//  which is assumed to be an instantaneous function of Vm
//  [NT] = NTmax / (1 + exp (-(Vm - Vp)/Kp))
//
// The gating of AMPAR is modeled using
//  Patneau - Mayer (1991) Neuron - Kinetic analysis of interaction between
//  Kainate and AMPA
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
#define Rb (getSharedMembers().Rb)
#define Ru1 (getSharedMembers().Ru1)
#define Ru2 (getSharedMembers().Ru2)
#define Rr (getSharedMembers().Rr)
#define Rd (getSharedMembers().Rd)

#define ALPHA (getSharedMembers().alpha)
#define BETA (getSharedMembers().beta)
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

void AMPAReceptor_Markov::initializeAMPA(RNG& rng)
{
#if SYNAPSE_MODEL_STRATEGY == USE_PRESYNAPTICPOINT
  assert(Vpre);
#endif
  //dyn_var_t ALPHANEUROTRANSMITTER = ALPHA * NEUROTRANSMITTER;
  //dyn_var_t r = ALPHANEUROTRANSMITTER / (BETA + ALPHANEUROTRANSMITTER);
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
}

void AMPAReceptor_Markov::updateAMPA(RNG& rng)
{
  fO = Tscale * (Ro * fC2 - Rc * fO) + fO;
  fC2 = Tscale * (Rb * NEUROTRANSMITTER * fC1 + Rc * fO + Rr * fD2 -
              (Ro + Ru2 + Rd) * fO) +
        fC2;
  fC1 = Tscale * (Rb * NEUROTRANSMITTER * fC0 + Ru2 * fC2 + Rr * fD1 -
              (Rb * NEUROTRANSMITTER + Ru1 + Rd) * fC1) +
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
}

void AMPAReceptor_Markov::setPostIndex(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_AMPAReceptor_MarkovInAttrPSet* CG_inAttrPset,
    CG_AMPAReceptor_MarkovOutAttrPSet* CG_outAttrPset)
{
  indexPost = CG_inAttrPset->idx;
#ifdef KEEP_PAIR_PRE_POST
  indexPrePost.push_back(&indexPost);
#endif
}

AMPAReceptor_Markov::~AMPAReceptor_Markov() {}
