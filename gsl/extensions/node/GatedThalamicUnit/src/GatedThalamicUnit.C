#include "Lens.h"
#include "GatedThalamicUnit.h"
#include "CG_GatedThalamicUnit.h"
#include "GridLayerData.h"
#include "NodeCompCategoryBase.h"
#include "rndm.h"
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void GatedThalamicUnit::initialize(RNG& rng) 
{
  double sumW = 0;
  ShallowArray<Input>::iterator iter, end=phenotypicInputs.end();
  for (iter=phenotypicInputs.begin(); iter!=end; ++iter)
    y0 += iter->weight * *(iter->input);
  ShallowArray<SpikeInput>::iterator iter2, end2=L5Inputs.end();
  for (iter2=L5Inputs.begin(); iter2!=end2; ++iter2)
    z0 += (*(iter2->spike)) ? iter2->weight:0.0;
}

void GatedThalamicUnit::update(RNG& rng) 
{
  // Phenotypic inputs
  double y = 0;
  ShallowArray<Input>::iterator iter, end=phenotypicInputs.end();
  for (iter=phenotypicInputs.begin(); iter!=end; ++iter) { 
    y += iter->weight * *(iter->input);
    y0 += SHD.betaY0 * (y - y0);
    y -= y0;
  }
  // L5 inputs (FFwd)
  double L5 = 0;
  ShallowArray<SpikeInput>::iterator iter2, end2=L5Inputs.end();
  for (iter2=L5Inputs.begin(); iter2!=end2; ++iter2)
    L5 += (*(iter2->spike)) ? iter2->weight:0.0;
  
  // TODO: insert here L5 inputs FB
  
  //Temporal averaging 
  z += SHD.alphaZ*(L5-z);
  z0 += SHD.betaZ0 * (z - z0);
  z -= z0;

  x = z+SHD.alpha*(y-z); // add phenotypic input

  // gating from striatum : must be moved up after L5 FFwd inputs
  double gate=0;
  ShallowArray<double*>::iterator iter3, end3=gateOpenInputs.end();
  for (iter3=gateOpenInputs.begin(); iter3!=end3; ++iter3) gate += (**iter3);
  end3=gateClosedInputs.end();
  for (iter3=gateClosedInputs.begin(); iter3!=end3; ++iter3) gate -= (**iter3);
  x *= (gate>=0) ? 1.0:0.0;
}

void GatedThalamicUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamicUnitInAttrPSet* CG_inAttrPset, CG_GatedThalamicUnitOutAttrPSet* CG_outAttrPset) 
{
  int row = getGlobalIndex()+1; // +1 is for Matlab
  int col = CG_node->getGlobalIndex()+1;

  if (CG_inAttrPset->identifier=="phenotypic") {
    phenotypicInputs[phenotypicInputs.size()-1].row = row;
    phenotypicInputs[phenotypicInputs.size()-1].col = col;
  }
  else assert(0);
}

void GatedThalamicUnit::outputWeights(std::ofstream& fsPH)
{
  ShallowArray<Input>::iterator PHiter,
    PHend=phenotypicInputs.end();

  for (PHiter=phenotypicInputs.begin(); PHiter!=PHend; ++PHiter)
    fsPH<<PHiter->row<<" "<<PHiter->col<<" "<<PHiter->weight<<std::endl;
}

void GatedThalamicUnit::inputWeight(std::ifstream& fsPH, int col)
{
  ShallowArray<Input>::iterator PHiter, PHend=phenotypicInputs.end();
  for (PHiter=phenotypicInputs.begin(); PHiter!=PHend; ++PHiter) {
    if (PHiter->col==col) {
      fsPH>>PHiter->weight;
      break;
    }
  }
}

GatedThalamicUnit::~GatedThalamicUnit() 
{
}

