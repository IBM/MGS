// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
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
  //std::cout << spikeInput.size() << std::endl;
  // Default starting values
  spike = 0;
  glutamate = 0.0;
  Cb1Rrise = 0.0;
  Cb1Rcurrent = 0.0;
}

void BoutonIAFUnit::update(RNG& rng)
{
  // ##### Glutamate release #####
  // If there is a spike, release glutamate
  if ((spikeInput.size() > 0) && (*(spikeInput[0].spike))) // only consider first one
    glutamate = availableGlutamate * spikeInput[0].weight; // weight is structural plasticity
  else
    glutamate = 0.0;



  // ##### Cb1R #####
  if (ECBinput.size() > 0)
    Cb1Rrise += ((-Cb1Rrise + (*(ECBinput[0].ECB) * ECBinput[0].weight)) // only consider first one, weight is structural plasticity
                 / SHD.Cb1RriseTau) * SHD.deltaT;
  else
    Cb1Rrise = 0.0;
  Cb1Rcurrent += ((-Cb1Rcurrent + Cb1Rrise) / SHD.Cb1RfallTau) * SHD.deltaT;



  // ##### Inhibit glutamate #####
  // Recovery glutamate
  availableGlutamate += ((maxGlutamate - availableGlutamate) / SHD.glutamateRecoverTau) * SHD.deltaT;
  // Inhibit the glutamate release with the activity of Cb1R
  availableGlutamate -= SHD.glutamateAdaptRate * Cb1Rcurrent;
  // Limit glutamate to >= 0
  if (availableGlutamate < 0.0)
    availableGlutamate = 0.0;
}

void BoutonIAFUnit::copy(RNG& rng)
{
  if (spikeInput.size() > 0)
    spike = *(spikeInput[0].spike); // only consider first one
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
  spikeInput[spikeInput.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab
  spikeInput[spikeInput.size()-1].col = CG_node->getGlobalIndex()+1;
}

void BoutonIAFUnit::setECBIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitOutAttrPSet* CG_outAttrPset)
{
  ECBinput[ECBinput.size()-1].row =  getGlobalIndex()+1; // +1 is for Matlab
  ECBinput[ECBinput.size()-1].col = CG_node->getGlobalIndex()+1;
}

BoutonIAFUnit::~BoutonIAFUnit()
{
}

