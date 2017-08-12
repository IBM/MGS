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
  // Default starting values
  AMPArise = 0.0;
  AMPAcurrent = 0.0;
  mGluR5rise = 0.0;
  mGluR5current = 0.0;
  Ca = 0.0;
  ECB = 0.0;
}

void SpineIAFUnit::update(RNG& rng)
{  
  // If the simulation has reached a certain period, apply a perturbation
  if (SHD.op_perturbation && ITER == SHD.perturbationT)
    AMPAweight = drandom(0.0, 1.5, rng);
  
  // ##### Vars needed #####
  double glutamate;
  if (glutamateInput.size() > 0)
    glutamate = *(glutamateInput[0].glutamate) * glutamateInput[0].weight; // only consider first one, weight is structural plasticity
  else
    glutamate = 0.0;



  // ##### AMPA #####
  // AMPA input is the minimum of AMPA weight and glutamate
  // Only updated when there is a pre-spike
  double AMPAinput = std::min(glutamate, AMPAweight);

  // Update AMPA rise with the AMPA activity
  AMPArise += ((-AMPArise + AMPAinput) / SHD.AMPAriseTau ) * SHD.deltaT;
  // Update AMPA current (fall) with the AMPA rise
  AMPAcurrent += ((-AMPAcurrent + AMPArise) / SHD.AMPAfallTau) * SHD.deltaT;



  // ##### mGluR5 #####
  // mGluR5 input is any excess glutamate bigger than AMPA weight
  double mGluR5input = 0.0;
  if (glutamate > AMPAweight)
    mGluR5input = (glutamate - AMPAweight) * SHD.mGluR5sensitivity; // adjust the sensitivity as well

  // Update mGluR5 rise with the mGluR5 activity
  mGluR5rise += ((-mGluR5rise + mGluR5input) / SHD.mGluR5riseTau ) * SHD.deltaT;
  // Update mGluR5 current (fall) with the mGluR5 rise
  mGluR5current += ((-mGluR5current + mGluR5rise) / SHD.mGluR5fallTau) * SHD.deltaT;

  

  // ##### Ca2+ #####
  // Ca2+ input
  double CaVSCCinput = 0.0;
  if (SHD.op_CaVSCCdepend)
    CaVSCCinput = SHD.CaVSCC * pow(AMPAweight, SHD.CaVSCCpow);
  else
    CaVSCCinput = SHD.CaVSCC;
  //  if (postSpikeInput.size() > 0)
  //    CaVSCCinput += (SHD.CaBP * (*(postSpikeInput[0].spike) * postSpikeInput[0].weight)); // only going to be one, weight is structural plasticity

  double CaInput = CaVSCCinput * AMPAcurrent;
  //  double CaInput = CaVSCCinput * (glutamate > 0.0 ? 1.0 : 0.0);

  // Update Ca2+ rise with VSCC and BP
  Carise += ((-Carise + CaInput) / SHD.CariseTau) * SHD.deltaT;
  // Update Ca2+ fall with Ca2+ rise
  Ca += ((-Ca + Carise) / SHD.CafallTau) * SHD.deltaT;



  // ##### endocannabinoids #####
  // Update ECB (is always in the range 0 to 1)
  // with Ca2+ and the mGluR5 modulation (AND gate)
  ECB = ECBproduction(Ca * mGluR5modulation(mGluR5current));
}

void SpineIAFUnit::outputWeights(std::ofstream& fs)
{
  float temp = (float) AMPAweight;
  fs.write(reinterpret_cast<char *>(&temp), sizeof(temp));
}

void SpineIAFUnit::setGlutamateIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitOutAttrPSet* CG_outAttrPset)
{
  glutamateInput[glutamateInput.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab
  glutamateInput[glutamateInput.size()-1].col = CG_node->getGlobalIndex()+1;
}

void SpineIAFUnit::setPostSpikeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitOutAttrPSet* CG_outAttrPset)
{
  postSpikeInput[postSpikeInput.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab
  postSpikeInput[postSpikeInput.size()-1].col = CG_node->getGlobalIndex()+1;
}

SpineIAFUnit::~SpineIAFUnit()
{
}

double SpineIAFUnit::ECBsigmoid(double Ca)
{
  return 1.0 / ( 1.0 + exp(-SHD.ECBprodC * (Ca - SHD.ECBprodD)) );
}

double SpineIAFUnit::ECBproduction(double Ca)
{
  // Computes the sigmoidal production of cannabinoids depending on Ca2+
  // NOTE: this is mirrored in SpineIAFUnitDataCollector. If changed here, change there too.
  double ECB = 0.0;
  // 1. the general sigmoid
  ECB = ECBsigmoid(Ca);
  // 2. make zero ECB at zero Ca2+
  ECB -= ECBsigmoid(0.0);
  // 3. Make one ECB at >= one Ca2+
  ECB *= 1.0 / (ECBsigmoid(1.0) - ECBsigmoid(0.0));
  if (ECB > 1.0)
    ECB = 1.0;

  return ECB;
}

double SpineIAFUnit::mGluR5modulation(double mGluR5)
{
  return ECBproduction(mGluR5); // just use the same modified sigmoid
}

