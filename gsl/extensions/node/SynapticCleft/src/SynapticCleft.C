#include "Lens.h"
#include "SynapticCleft.h"
#include "CG_SynapticCleft.h"
#include "rndm.h"

#include "MaxComputeOrder.h"

//#define	Glut_max  *(getSharedMembers().Glut_baseline);
//#define	GABA_max  *(getSharedMembers().GABA_baseline);
#define numNTperVesicle 100000
// NOTE: Upto 130 vesicles released per bouton over 1-min period of 0.2Hz stimulus [Ikeda-Bekkers, 2009]
// 3 groups of synaptic vesicles: RRP, RP, NRP
//  Synaptic cleft volume 
//  V = length * width * depth
//   length = active zone diameter = 300+/- 150 nm
//   width = cleft width = 20 +/- 2.8 nm
//   depth = width = 20

void SynapticCleft::produceInitialState(RNG& rng) 
{
  if (branchDataConnect) branchData=*branchDataConnect;
	Glut = (getSharedMembers().Glut_baseline);
	GABA = (getSharedMembers().GABA_baseline);
//#if GLUTAMATE_UPDATE_METHOD == NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994 || \
//	GLUTAMATE_UPDATE_METHOD ==  NEUROTRANSMITTER_BIEXPONENTIAL
//  Glut = 0.0;
//#endif
//#if GABA_UPDATE_METHOD == NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994 || \
//	GABA_UPDATE_METHOD ==  NEUROTRANSMITTER_BIEXPONENTIAL
//	GABA = 0.0;
//#endif
}

void SynapticCleft::produceState(RNG& rng) 
{
	dyn_var_t* V = Vpre;
	{//Glut
#if GLUTAMATE_UPDATE_METHOD == NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
		Glut =  (getSharedMembers().Glut_max/(1.0 + exp(-(*V - getSharedMembers().Vp_Glut)/getSharedMembers().Kp_Glut)));
#elif GLUTAMATE_UPDATE_METHOD == NEUROTRANSMITTER_BIEXPONENTIAL
		dyn_var_t dt = *(getSharedMembers().deltaT);
		float J_decay = 1.0 / ((getSharedMembers().tau_Glut)) * (
												Glut - getSharedMembers().Glut_baseline);  // [uM/msec]
		Glut = Glut + dt *( 
				(getSharedMembers().Glut_max/(1.0 + exp(-(*V - getSharedMembers().Vp_Glut)/getSharedMembers().Kp_Glut)))
				- J_decay);
#else
		 // may be dynamics true concentration of Glut
		NOT IMPLEMENTED YET
#endif
	}
	{//GABA
#if GABA_UPDATE_METHOD == NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
		GABA =  (getSharedMembers().GABA_max/(1.0 + exp(-(*V - getSharedMembers().Vp_GABA)/getSharedMembers().Kp_GABA)));
#elif GABA_UPDATE_METHOD == NEUROTRANSMITTER_BIEXPONENTIAL
		dyn_var_t dt = *(getSharedMembers().deltaT);
		float J_decay = 1.0 / ((getSharedMembers().tau_GABA)) * (
												GABA - getSharedMembers().GABA_baseline);  // [uM/msec]
		GABA = GABA + dt * ( 
				(getSharedMembers().GABA_max/(1.0 + exp(-(*V - getSharedMembers().Vp_GABA)/getSharedMembers().Kp_GABA)))
				- J_decay);
#else
		NOT IMPLEMENTED YET
#endif
	}
}

void SynapticCleft::setPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SynapticCleftInAttrPSet* CG_inAttrPset, CG_SynapticCleftOutAttrPSet* CG_outAttrPset) 
{
  index=CG_inAttrPset->idx;
  assert(getSharedMembers().voltageConnect);
  assert(index>=0 && index<getSharedMembers().voltageConnect->size());    
  Vpre = &((*(getSharedMembers().voltageConnect))[index]);
}

SynapticCleft::~SynapticCleft() 
{
}

