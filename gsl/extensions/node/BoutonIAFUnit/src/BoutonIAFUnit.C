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
#include "BoutonIAFUnit.h"
#include "CG_BoutonIAFUnit.h"
#include "rndm.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()

void BoutonIAFUnit::initialize(RNG& rng)
{
  // Check if more than one input
  if (spikeInput.size() != 1)
    assert("BoutonIAFUnit: spike inputs should be one.");
  if (ECBinput.size() != 1)
    assert("BoutonIAFUnit: ECB inputs should be one.");
  if (CB1input.size() != 1)
    assert("BoutonIAFUnit: Goodwin inputs should be one.");
  // Default starting values
  neurotransmitter = 0.0;
  CB1Rrise = 0.0;
  CB1Rcurrent = 0.0;
}

void BoutonIAFUnit::update(RNG& rng)
{
  // ##### Neurotransmitter release #####
  // If there is a spike, release neurotransmitter
  if ((spikeInput.size() > 0) && (*(spikeInput[0].spike))) // only consider first one
    neurotransmitter = availableNeurotransmitter * spikeInput[0].weight; // weight is structural plasticity
  else
    neurotransmitter = 0.0;



  // ##### CB1R #####
  if (CB1input.size() > 0)
    {
      CB1R = (*(CB1input[0].Y) * CB1input[0].weight); // only consider first one, weight is scaling from Goodwin model Y
      if (CB1R > 1.0) // bound just in case
        CB1R = 1.0;
      else if (CB1R < 0.0)
        CB1R = 0.0;
    }
  ECB = 0.0;
  if (ECBinput.size() > 0)
    {
      ECB = (*(ECBinput[0].ECB) * ECBinput[0].weight); // only consider first one, weight is structural plasticity
      CB1Runbound = CB1R - ECB;
      if (CB1Runbound < 0.0)
        CB1Runbound = 0.0;
      CB1Rrise += ((-CB1Rrise + ECB) / SHD.CB1RriseTau) * SHD.deltaT;
    }
  else
    {
      CB1Runbound = CB1R;
      CB1Rrise = 0.0;
    }
  CB1Rcurrent += ((-CB1Rcurrent + CB1Rrise) / SHD.CB1RfallTau) * SHD.deltaT;



  // ##### Inhibit neurotransmitter #####
  // Recovery neurotransmitter
  availableNeurotransmitter += ((maxNeurotransmitter - availableNeurotransmitter) / SHD.neurotransmitterRecoverTau) * SHD.deltaT;
  // Inhibit the neurotransmitter release with the quantity of ECB and CB1R, i.e. the minimum
  availableNeurotransmitter -= SHD.neurotransmitterAdaptRate * std::min(ECB, CB1R);
  // Limit neurotransmitter to >= 0
  if (availableNeurotransmitter < 0.0)
    availableNeurotransmitter = 0.0;
}

void BoutonIAFUnit::outputIndexs(std::ofstream& fs)
{
  ShallowArray<SpikeInput>::iterator iter, end=spikeInput.end();
  if (spikeInput.size() > 0)
    {
      int col = spikeInput[0].col;
      fs.write(reinterpret_cast<char *>(&col), sizeof(col));
      int row = spikeInput[0].row;
      fs.write(reinterpret_cast<char *>(&row), sizeof(row));
    }
  else
    {
      int col = -9;
      fs.write(reinterpret_cast<char *>(&col), sizeof(col));
      int row = -9;
      fs.write(reinterpret_cast<char *>(&row), sizeof(row));
    }
}

void BoutonIAFUnit::setSpikeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitOutAttrPSet* CG_outAttrPset)
{
  spikeInput[spikeInput.size()-1].row =  getIndex()+1; // +1 is for Matlab
  spikeInput[spikeInput.size()-1].col = CG_node->getIndex()+1;
}

void BoutonIAFUnit::setECBIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitOutAttrPSet* CG_outAttrPset)
{
  ECBinput[ECBinput.size()-1].row =  getIndex()+1; // +1 is for Matlab
  ECBinput[ECBinput.size()-1].col = CG_node->getIndex()+1;
}

BoutonIAFUnit::~BoutonIAFUnit()
{
}

