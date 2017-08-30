// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "SpineIAFUnit.h"
#include "CG_SpineIAFUnit.h"
#include "rndm.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void SpineIAFUnit::initialize(RNG& rng)
{
  // Check if more than one input
  if (neurotransmitterInput.size() != 1)
    assert("SpineIAFUnit: neurotransmitter inputs should be one.");
  if (postSpikeInput.size() != 1)
    assert("SpineIAFUnit: post-synaptic spike inputs should be one.");
  // Default starting values
  AMPArise = 0.0;
  AMPAcurrent = 0.0;
  mGluR5rise = 0.0;
  mGluR5current = 0.0;
  Ca = 0.0;
  eCB = 0.0;
}

void SpineIAFUnit::update(RNG& rng)
{  
  // If the simulation has reached a certain period, apply a perturbation
  if (SHD.op_perturbation && ITER == SHD.perturbationT)
    AMPAweight = drandom(0.0, 1.5, rng);
  
  // ##### Vars needed #####
  double neurotransmitter;
  if (neurotransmitterInput.size() > 0)
    neurotransmitter = *(neurotransmitterInput[0].neurotransmitter) * neurotransmitterInput[0].weight; // only consider first one, weight is structural plasticity
  else
    neurotransmitter = 0.0;



  // ##### AMPA #####
  // AMPA input is the minimum of AMPA weight and neurotransmitter
  // Only updated when there is a pre-spike
  double AMPAinput = std::min(neurotransmitter, AMPAweight);

  // Update AMPA rise with the AMPA activity
  AMPArise += ((-AMPArise + AMPAinput) / SHD.AMPAriseTau ) * SHD.deltaT;
  // Update AMPA current (fall) with the AMPA rise
  AMPAcurrent += ((-AMPAcurrent + AMPArise) / SHD.AMPAfallTau) * SHD.deltaT;



  // ##### mGluR5 #####
  // mGluR5 input is any excess neurotransmitter bigger than AMPA weight
  double mGluR5input = 0.0;
  if (neurotransmitter > AMPAweight)
    mGluR5input = (neurotransmitter - AMPAweight) * SHD.mGluR5sensitivity; // adjust the sensitivity as well

  // Update mGluR5 rise with the mGluR5 activity
  mGluR5rise += ((-mGluR5rise + mGluR5input) / SHD.mGluR5riseTau ) * SHD.deltaT;
  // Update mGluR5 current (fall) with the mGluR5 rise
  mGluR5current += ((-mGluR5current + mGluR5rise) / SHD.mGluR5fallTau) * SHD.deltaT;



  // ##### NMDAR #####
  double NMDARopenInput = 0.0;
  if (neurotransmitterInput.size() > 0)
    NMDARopenInput = (*(neurotransmitterInput[0].neurotransmitter) * neurotransmitterInput[0].weight); // only going to be one, weight is structural plasticity; adjust the sensitivity as well
  NMDARopen += ((-NMDARopen + NMDARopenInput) / SHD.NMDARopenTau) * SHD.deltaT;
  NMDARCarise += ((-NMDARCarise + ((*(postSpikeInput[0].spike) * postSpikeInput[0].weight) // only going to be one, weight is structural plasticity
                                   * NMDARopen * SHD.NMDARCasensitivity))
                  / SHD.NMDARCariseTau) * SHD.deltaT;
  NMDARCacurrent += ((-NMDARCacurrent + NMDARCarise) / SHD.NMDARCafallTau) * SHD.deltaT;  

  

  // ##### Ca2+ #####
  // Ca2+ input
  double CaVSCCinput = 0.0;
  if (SHD.op_CaVSCCdepend)
    CaVSCCinput = SHD.CaVSCC * pow(AMPAweight, SHD.CaVSCCpow);
  else
    CaVSCCinput = SHD.CaVSCC;
  double CaInput = (CaVSCCinput * AMPAcurrent) + NMDARCacurrent;

  // Update Ca2+ rise with VSCC and BP
  Carise += ((-Carise + CaInput) / SHD.CariseTau) * SHD.deltaT;
  // Update Ca2+ fall with Ca2+ rise
  Ca += ((-Ca + Carise) / SHD.CafallTau) * SHD.deltaT;



  // ##### endocannabinoids #####
  // Update eCB (is always in the range 0 to 1)
  // with Ca2+ and the mGluR5 modulation (AND gate)
  eCB = eCBproduction(Ca * mGluR5modulation(mGluR5current));
}

void SpineIAFUnit::outputWeights(std::ofstream& fs)
{
  float temp = (float) AMPAweight;
  fs.write(reinterpret_cast<char *>(&temp), sizeof(temp));
}

void SpineIAFUnit::setNeurotransmitterIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitOutAttrPSet* CG_outAttrPset)
{
  neurotransmitterInput[neurotransmitterInput.size()-1].row =  getIndex()+1; // +1 is for Matlab
  neurotransmitterInput[neurotransmitterInput.size()-1].col = CG_node->getIndex()+1;
}

void SpineIAFUnit::setPostSpikeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitOutAttrPSet* CG_outAttrPset)
{
  postSpikeInput[postSpikeInput.size()-1].row =  getIndex()+1; // +1 is for Matlab
  postSpikeInput[postSpikeInput.size()-1].col = CG_node->getIndex()+1;
}

SpineIAFUnit::~SpineIAFUnit()
{
}

double SpineIAFUnit::eCBsigmoid(double Ca)
{
  return 1.0 / ( 1.0 + exp(-SHD.eCBprodC * (Ca - SHD.eCBprodD)) );
}

double SpineIAFUnit::eCBproduction(double Ca)
{
  // Computes the sigmoidal production of cannabinoids depending on Ca2+
  // NOTE: this is mirrored in SpineIAFUnitDataCollector. If changed here, change there too.
  double eCB = 0.0;
  // 1. the general sigmoid
  eCB = eCBsigmoid(Ca);
  // 2. make zero eCB at zero Ca2+
  eCB -= eCBsigmoid(0.0);
  // 3. Make one eCB at >= one Ca2+
  eCB *= 1.0 / (eCBsigmoid(1.0) - eCBsigmoid(0.0));
  if (eCB > 1.0)
    eCB = 1.0;

  return eCB;
}

double SpineIAFUnit::mGluR5modulation(double mGluR5)
{
  return eCBproduction(mGluR5); // just use the same modified sigmoid
}

