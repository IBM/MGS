#include "Lens.h"
#include "GatedThalamoCorticalUnit.h"
#include "CG_GatedThalamoCorticalUnit.h"
#include "GridLayerData.h"
#include "NodeCompCategoryBase.h"
#include "rndm.h"
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void GatedThalamoCorticalUnit::initialize(RNG& rng) 
{
  double sumW = 0;
  ShallowArray<Input>::iterator iter, end=phenotypicInputs.end();
  for (iter=phenotypicInputs.begin(); iter!=end; ++iter)
    y0 += iter->weight * *(iter->input);
  ShallowArray<SpikeInput>::iterator iter2, end2=L5FFInputs.end();
  for (iter2=L5FFInputs.begin(); iter2!=end2; ++iter2)
    z0 += (*(iter2->spike)) ? iter2->weight:0.0;
  ShallowArray<SpikeInput>::iterator iter3, end3=L5FBInputs.end();
  for (iter3=L5FBInputs.begin(); iter3!=end3; ++iter3)
    z0 += (*(iter3->spike)) ? iter3->weight:0.0;
}

void GatedThalamoCorticalUnit::update(RNG& rng) 
{
  // Phenotypic inputs
  y = 0;
  ShallowArray<Input>::iterator iter, end=phenotypicInputs.end();
  for (iter=phenotypicInputs.begin(); iter!=end; ++iter) { 
    y += iter->weight * *(iter->input);
    //y0 += SHD.betaY0 * (y - y0);
    //y -= y0;
  }
  
  // Lateral inhibition 
  ShallowArray<Input>::iterator iter5, end5=RF_Inputs.end();
  for (iter5=RF_Inputs.begin(); iter5!=end5; ++iter5) {
    y -= (*(iter5->input) * iter5->weight)>0 ? (*(iter5->input) * iter5->weight):0.0;
  }
  y *= (y+1>0) ? 1.0:0.0;
  
  // L5 inputs (FFwd)
  double L5FF = 0;
  ShallowArray<SpikeInput>::iterator iter2, end2=L5FFInputs.end();
  for (iter2=L5FFInputs.begin(); iter2!=end2; ++iter2)
    L5FF += (*(iter2->spike)) ? iter2->weight:0.0;
  
  // L5 FFwd inputs gating from striatum
  double gate=1; // set gate=1 so that no gating lets everything pass 
  ShallowArray<double*>::iterator iter3, end3=gateOpenInputs.end();
  for (iter3=gateOpenInputs.begin(); iter3!=end3; ++iter3) gate += (**iter3);
  end3=gateClosedInputs.end();
  for (iter3=gateClosedInputs.begin(); iter3!=end3; ++iter3) gate -= (**iter3);
  L5FF *= (gate>=0) ? 1.0:0.0;

  // L5 inputs (FBck)
  double L5FB = 0;
  ShallowArray<SpikeInput>::iterator iter4, end4=L5FBInputs.end();
  for (iter4=L5FBInputs.begin(); iter4!=end4; ++iter4)
    L5FB += (*(iter4->spike)) ? iter4->weight:0.0;

  //x = z+SHD.alpha*(y-z); // TODO:add phenotypic input here
  
  // Temporal averaging 
  z += SHD.alphaZ*(L5FF*(y+1)+L5FB-z);
  z0 += SHD.betaZ0 * (z - z0);
  z -= z0;
}

void GatedThalamoCorticalUnit::whiten(RNG& rng){
  if (SHD.whitening) {
    // update the value of the covariance between units (which is stored in weights) and set whitened output x
    ShallowArray<GTCU_LN_Input>::iterator iter, end=lateralInputs.end();
    x=0;
    for (iter=lateralInputs.begin(); iter!=end; ++iter) {
      iter->covWeight += SHD.betaZ0 * (z * *(iter->input));
      x += *(iter->input) * iter->whitWeight;
    }
  } else {
    x=z;
  }
}

void GatedThalamoCorticalUnit::setIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_GatedThalamoCorticalUnitInAttrPSet* CG_inAttrPset, CG_GatedThalamoCorticalUnitOutAttrPSet* CG_outAttrPset) 
{
  int row = getGlobalIndex()+1; // +1 is for Matlab
  int col = CG_node->getGlobalIndex()+1;

  if (CG_inAttrPset->identifier=="phenotypic") {
    phenotypicInputs[phenotypicInputs.size()-1].row = row;
    phenotypicInputs[phenotypicInputs.size()-1].col = col;
  }
  else if (CG_inAttrPset->identifier=="LN_GTCU") {
    lateralInputs[lateralInputs.size()-1].row = row;
    lateralInputs[lateralInputs.size()-1].col = col;
  } 
  else if (CG_inAttrPset->identifier=="RF_GTCU") {
    RF_Inputs[RF_Inputs.size()-1].row = row;
    RF_Inputs[RF_Inputs.size()-1].col = col;
  }
  else assert(0);
}

void GatedThalamoCorticalUnit::outputWeights(std::ofstream& fsPH)
{
  ShallowArray<Input>::iterator PHiter,
    PHend=phenotypicInputs.end();

  for (PHiter=phenotypicInputs.begin(); PHiter!=PHend; ++PHiter)
    fsPH<<PHiter->row<<" "<<PHiter->col<<" "<<PHiter->weight<<std::endl;
}

void GatedThalamoCorticalUnit::inputWeight(std::ifstream& fsPH, int col)
{
  ShallowArray<Input>::iterator PHiter, PHend=phenotypicInputs.end();
  for (PHiter=phenotypicInputs.begin(); PHiter!=PHend; ++PHiter) {
    if (PHiter->col==col) {
      fsPH>>PHiter->weight;
      break;
    }
  }
}

void GatedThalamoCorticalUnit::getLateralCovInputs(std::ofstream& fsLN) 
{
  ShallowArray<GTCU_LN_Input>::iterator it;
  ShallowArray<GTCU_LN_Input>::iterator end = lateralInputs.end();
  for (it=lateralInputs.begin(); it!=end; ++it) {
    fsLN << it->covWeight << " ";
  }
  fsLN << std::endl;
}

void GatedThalamoCorticalUnit::setLateralWhitInputs(std::vector<double>* latWhitInputs) 
{
  ShallowArray<GTCU_LN_Input>::iterator it;
  ShallowArray<GTCU_LN_Input>::iterator end = lateralInputs.end();
  std::vector<double>& lWhitInputs = *latWhitInputs;
  int idx=0;
  for (it=lateralInputs.begin(); it!=end; ++it) {
    it->whitWeight = lWhitInputs[idx];
    it->covWeight = 0; //reset covariance matrix at each time the whitening matrix is updated
    idx++;
  } 
}

GatedThalamoCorticalUnit::~GatedThalamoCorticalUnit() 
{
}

