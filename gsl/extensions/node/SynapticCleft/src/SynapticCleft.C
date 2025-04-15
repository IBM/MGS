#include "CG_SynapticCleft.h"
#include "Lens.h"
#include "SynapticCleft.h"
#include "rndm.h"
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
*/

#include "MaxComputeOrder.h"
#include "NodeProxyBase.h"

//#define	Glut_max  *(getSharedMembers().Glut_baseline);
//#define	GABA_max  *(getSharedMembers().GABA_baseline);
//#define numNTperVesicle 100000
//#define numNTperVesicle 3000
// NOTE: Upto 130 vesicles released per bouton over 1-min period of 0.2Hz
// stimulus [Ikeda-Bekkers, 2009]
// 3 groups of synaptic vesicles: RRP, RP, NRP
//  Synaptic cleft volume
//  V = length * width * depth
//   length = active zone diameter = 300+/- 150 nm
//   width = cleft width = 20 +/- 2.8 nm
//   depth = width = 20
#define Vthreshold 0.0
//#define Cleft_Volume (0.76*1e-3)    // um^3
//#define Cleft_Volume (0.76*1e-3*1e-15)    // Litre
//#define Cleft_Volume (2.16e-3*1e-15)    // Litre - effective volume
#define Cleft_Volume (getSharedMembers().cleftVolume * 1e-15)    // Litre - effective volume

#define GlutperVesicle (getSharedMembers().numGlutperVesicle)
#define GABAperVesicle (getSharedMembers().numGABAperVesicle)

#define Glut_max (getSharedMembers().Glut_max)
#define Vp_Glut (getSharedMembers().Vp_Glut)
#define Kp_Glut (getSharedMembers().Kp_Glut)
#define dt (*(getSharedMembers().deltaT))
#define GABA_max (getSharedMembers().GABA_max)
#define Vp_GABA (getSharedMembers().Vp_GABA)
#define Kp_GABA (getSharedMembers().Kp_GABA)

void SynapticCleft::produceInitialState(RNG& rng)
{
  Glut = (getSharedMembers().Glut_baseline);
  GABA = (getSharedMembers().GABA_baseline);
#if SUPPORT_MODULABLE_CLEFT
  DA = (getSharedMembers().DA_baseline);
  Ser = (getSharedMembers().Ser_baseline);
#endif
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
  if (Vpre == NULL)
    //dummy SynapticCleft
    return;
  float currentTime = dt * (getSimulation().getIteration());
  if (*V < Vthreshold)
  {
    _reset = true;
  }

  {  // Glut
#if GLUTAMATE_UPDATE_METHOD == NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
    Glut = (Glut_max / (1.0 + exp(-(*V - Vp_Glut) / Kp_Glut)));
#elif GLUTAMATE_UPDATE_METHOD == NEUROTRANSMITTER_BIEXPONENTIAL
    float J_decay = 1.0 / ((getSharedMembers().tau_Glut)) *
                    (Glut - getSharedMembers().Glut_baseline);  // [uM/msec]
    //Glut = Glut +
    //        ((Glut_max / (1.0 + exp(-(*V - Vp_Glut) / Kp_Glut))) - dt *J_decay);
    //       //dt * ((Glut_max / (1.0 + exp(-(*V - Vp_Glut) / Kp_Glut))) - J_decay);
   Glut -= dt * J_decay;
#else
    // may be dynamics true concentration of Glut
    NOT IMPLEMENTED YET
#endif
  }
  {  // GABA
#if GABA_UPDATE_METHOD == NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
    GABA = (GABA_max /
            (1.0 + exp(-(*V - Vp_GABA) /
                       Kp_GABA)));
#elif GABA_UPDATE_METHOD == NEUROTRANSMITTER_BIEXPONENTIAL
    float J_decay = 1.0 / ((getSharedMembers().tau_GABA)) *
                    (GABA - getSharedMembers().GABA_baseline);  // [uM/msec]
    //GABA = GABA +
    //       dt * (GABA_max /
    //              (1.0 + exp(-(*V - getSharedMembers().Vp_GABA) /
    //                         getSharedMembers().Kp_GABA))) -
    //             J_decay);
    GABA -= dt * J_decay;
    //GABA = GABA +
    //  (GABA_max /
    //    (1.0 + exp(-(*V - getSharedMembers().Vp_GABA) /
    //    	   getSharedMembers().Kp_GABA))) -
    //   dt * J_decay);
   
#else
    NOT IMPLEMENTED YET
#endif
  }
#if SUPPORT_MODULABLE_CLEFT
  {// DA
    assert(0);
  } 
  {// Ser
    assert(0);
  }
#endif
  if (*V > Vthreshold and _reset and
      (currentTime - _timeLastSpike > 0.01))
  {
    //NOTE: using _timeLastSpike --> goal: to use that to prevent
    //further
    //release at two adjacent spikes or variation in Vm
    _timeLastSpike =  dt * (getSimulation().getIteration());
    _reset = false;
    Glut = (Glut_max / (1.0 + exp(-(*V - Vp_Glut) / Kp_Glut)));
#if GLUTAMATE_UPDATE_METHOD == NEUROTRANSMITTER_BIEXPONENTIAL
    Glut += (GlutperVesicle / AvogN) / (Cleft_Volume) * 1e6;
#endif
    GABA = (GABA_max / (1.0 + exp(-(*V - Vp_GABA) / Kp_GABA)));
#if GABA_UPDATE_METHOD == NEUROTRANSMITTER_BIEXPONENTIAL
    GABA += (GABAperVesicle / AvogN) / (Cleft_Volume) * 1e6;
#endif
  }
}

void SynapticCleft::setPointers(const String& CG_direction,
                                const String& CG_component,
                                NodeDescriptor* CG_node, Edge* CG_edge,
                                VariableDescriptor* CG_variable,
                                Constant* CG_constant,
                                CG_SynapticCleftInAttrPSet* CG_inAttrPset,
                                CG_SynapticCleftOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG
  data_received.append(CG_inAttrPset->side.c_str());
#endif
  if (CG_inAttrPset->side == "pre")
  {
    int index = CG_inAttrPset->idx;
    assert(getSharedMembers().voltageConnect);
    assert(index >= 0 && index < getSharedMembers().voltageConnect->size());
    Vpre = &((*(getSharedMembers().voltageConnect))[index]);
    dimensionsPrePost.push_back((*(getSharedMembers().dimensionsArray_connect))[index]);

#ifdef KEEP_PAIR_PRE_POST
    indexPrePost.push_back(index);
#else
    if (getSharedMembers().branchDataConnect)
      branchDataPre = *(getSharedMembers().branchDataConnect);
    assert(0);  // not completed yet for this mode of macro
#endif
  }
  else if (CG_inAttrPset->side == "post")
  {
    //wrong: int index = CG_inAttrPset->idx + getSharedMembers().voltageConnect->size();
    int index = CG_inAttrPset->idx;
    NodeProxyBase* node = dynamic_cast<NodeProxyBase*>(CG_node->getNode());
    if (node == 0)
    {//not a proxy
      dimensionsPrePost.push_back((*(getSharedMembers().dimensionsArray_connect))[index]);
    }
    else
    {
      //TUAN TODO: this is a temporary workaround solution
      // i.e. use the location of the pre as the post 
      // we need this to use for connection time 
      dimensionsPrePost.push_back(dimensionsPrePost[0]);
    }
#ifdef KEEP_PAIR_PRE_POST
    indexPrePost.push_back(index);
#else
    if (getSharedMembers().branchDataConnect)
      branchDataPost = *(getSharedMembers().branchDataConnect);
#endif
  }
  else
  {
    assert(0);
  }
}

SynapticCleft::~SynapticCleft() {}
