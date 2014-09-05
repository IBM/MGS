// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CaConcentrationJunction.h"
#include "CG_CaConcentrationJunction.h"
#include "rndm.h"
#include "GridLayerDescriptor.h"
#define DISTANCE_SQUARED(a,b) ((((a).x-(b).x)*((a).x-(b).x))+(((a).y-(b).y)*((a).y-(b).y))+(((a).z-(b).z)*((a).z-(b).z)))
#define uM_um_cubed_per_pA_msec 5.18213484752067
//#define DEBUG_HH

#ifdef DEBUG_HH
#include "../../../../../nti/SegmentDescriptor.h"
#endif

float CaConcentrationJunction::getArea(DimensionStruct* a, DimensionStruct* b) {
  float radius = 0.5*(b->r + 0.5*(a->r + b->r));
  float length = 0.5*sqrt(DISTANCE_SQUARED(*a, *b));
  return(2.0*M_PI*radius*length);
}

float CaConcentrationJunction::getArea() {
  float area=0.0;
  assert(dimensions.size()==1);
  DimensionStruct* dimension=dimensions[0];
  Array<DimensionStruct*>::iterator iter=dimensionInputs.begin(), end=dimensionInputs.end(); 
  for (; iter!=end; ++iter) {
    area += getArea(*iter, dimension);
  }
  return area;
}

void CaConcentrationJunction::initializeJunction(RNG& rng) 
{
  assert(dimensions.size()==1);
  DimensionStruct* dimension=dimensions[0];
  assert(Ca_new.size()==1);
  Ca_cur=Ca_new[0];
  Array<DimensionStruct*>::iterator iter=dimensionInputs.begin(), end=dimensionInputs.end(); 
  for (; iter!=end; ++iter) {
    
    float R = 0.5 * (0.5 * ( (*iter)->r + dimension->r ) + dimension->r);
    float L = 0.5 * sqrt(DISTANCE_SQUARED(**iter, *dimension));
    
    volume += M_PI * R * R * L;
    //std::cerr<<"volume:"<<volume<<std::endl<<std::endl;
  }
  float Pdov = M_PI * getSharedMembers().DCa / volume;
  currentToConc = getArea() * uM_um_cubed_per_pA_msec / volume;

  Array<DimensionStruct*>::iterator diter=dimensionInputs.begin(), dend=dimensionInputs.end(); 
  for (; diter!=dend; ++diter) {
    float Rb = 0.5 * ( (*diter)->r + dimension->r );
    fAxial.push_back(Pdov * Rb * Rb / sqrt(DISTANCE_SQUARED(**diter, *dimension)));
  }
#ifdef DEBUG_HH
  std::cerr<<"CA_JUNCTION ("<<dimension->x<<","<<dimension->y<<","<<dimension->z<<","<<dimension->r<<")"<<std::endl;
#endif
}

void CaConcentrationJunction::predictJunction(RNG& rng) 
{
  assert(getSharedMembers().bmt>0);
  float LHS = getSharedMembers().bmt;
  float RHS = (getSharedMembers().bmt - getSharedMembers().CaClearance) * Ca_cur;

  Array<ChannelCaCurrents>::iterator citer=channelCaCurrents.begin();
  Array<ChannelCaCurrents>::iterator cend=channelCaCurrents.end();
  for (; citer!=cend; ++citer) {
    RHS -= currentToConc * (*(citer->currents))[0];
  }

  Array<float*>::iterator iter=receptorCaCurrents.begin();
  Array<float*>::iterator end=receptorCaCurrents.end();
  for (; iter!=end; ++iter) {
    RHS -= currentToConc * **iter;
  }

  iter=injectedCaCurrents.begin();
  end=injectedCaCurrents.end();
  for (; iter!=end; ++iter) {
    RHS += **iter * currentToConc / getArea();
  }

  Array<float>::iterator xiter=fAxial.begin(), xend=fAxial.end(); 
  Array<float*>::iterator viter=CaConcentrationInputs.begin();
  for (; xiter!=xend; ++xiter, ++viter) {
    RHS += (*xiter) * ((**viter) - Ca_cur);
  }

  Ca_new[0] = RHS/LHS;

#ifdef DEBUG_HH
  std::cerr<<getSimulation().getIteration() * *getSharedMembers().deltaT
	   <<" CA_JUNCTION PREDICT"
	   <<" ["<<getSimulation().getRank()<<","<<getNodeIndex()<<","
	   <<getIndex()<<"] "
	   <<"("<<segmentDescriptor.getNeuronIndex(branchData->key)
	   <<","<<segmentDescriptor.getBranchIndex(branchData->key)
	   <<","<<segmentDescriptor.getBranchOrder(branchData->key)
	   <<") {"
	   <<dimension->x<<","<<dimension->y<<","<<dimension->z<<","<<dimension->r<<"} "
	   <<Ca_new[0]<<std::endl;
#endif
}

void CaConcentrationJunction::correctJunction(RNG& rng) 
{
  assert(getSharedMembers().bmt>0);
  float LHS = getSharedMembers().bmt;
  float RHS = (getSharedMembers().bmt - getSharedMembers().CaClearance) * Ca_cur;

  Array<ChannelCaCurrents>::iterator citer=channelCaCurrents.begin();
  Array<ChannelCaCurrents>::iterator cend=channelCaCurrents.end();
  for (; citer!=cend; ++citer) {
    RHS -= currentToConc * (*(citer->currents))[0];
  }

  Array<float*>::iterator iter=receptorCaCurrents.begin();
  Array<float*>::iterator end=receptorCaCurrents.end();
  for (; iter!=end; ++iter) {
    RHS -= currentToConc * **iter;
  }

  iter=injectedCaCurrents.begin();
  end=injectedCaCurrents.end();
  for (; iter!=end; ++iter) {
    RHS += **iter * currentToConc / getArea();
  }

  Array<float>::iterator xiter=fAxial.begin(), xend=fAxial.end(); 
  Array<float*>::iterator viter=CaConcentrationInputs.begin();
  for (; xiter!=xend; ++xiter, ++viter) {
    LHS += (*xiter);
    RHS += (*xiter) * (**viter);
  }

  Ca_new[0] = RHS/LHS;

  // This is the swap phase
  Ca_cur = Ca_new[0] = 2.0 * Ca_new[0] - Ca_cur;

#ifdef DEBUG_HH
  assert(dimensions.size()==1);
  DimensionStruct* dimension=dimensions[0];
  std::cerr<<getSimulation().getIteration() * *getSharedMembers().deltaT
	   <<" CA_JUNCTION CORRECT"
	   <<" ["<<getSimulation().getRank()<<","<<getNodeIndex()<<","
	   <<getIndex()<<"] "
	   <<"("<<segmentDescriptor.getNeuronIndex(branchData->key)
	   <<","<<segmentDescriptor.getBranchIndex(branchData->key)
	   <<","<<segmentDescriptor.getBranchOrder(branchData->key)
	   <<") {"
	   <<dimension->x<<","<<dimension->y<<","<<dimension->z<<","<<dimension->r<<"} "
	   <<Ca_new[0]<<std::endl;

  Array<DimensionStruct*>::iterator diter=dimensionInputs.begin();
  Array<float*>::iterator vend=CaConcentrationInputs.end();
  int c=0;

  for (viter=CaConcentrationInputs.begin(); viter!=vend; ++viter, ++diter) {
    std::cerr<<getSimulation().getIteration() * *getSharedMembers().deltaT
               <<" CA_JCT_INPUT_"<<c++
	       <<" ["<<getSimulation().getRank()<<","<<getNodeIndex()<<","
	       <<getIndex()<<"] "
	       <<"("<<segmentDescriptor.getNeuronIndex(branchData->key)
	       <<","<<segmentDescriptor.getBranchIndex(branchData->key)
	       <<","<<segmentDescriptor.getBranchOrder(branchData->key)
	       <<","<<segmentDescriptor.getComputeOrder(branchData->key)
	       <<") {"
	       <<(*diter)->x<<","<<(*diter)->y<<","<<(*diter)->z<<","<<(*diter)->r<<"} "
	       <<DISTANCE_SQUARED(*(*diter), *dimension)<<" "
	       <<*(*viter)<<std::endl;
  }
#endif
}

bool CaConcentrationJunction::checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset) 
{
  assert(dimensions.size()==1);
  DimensionStruct* dimension=dimensions[0];
  TissueSite& site=CG_inAttrPset->site;
  bool rval=(site.r==0);
  if (!rval)
    rval=( (site.r*site.r) >= DISTANCE_SQUARED(site, *dimension) );
  return rval;
}

bool CaConcentrationJunction::confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset) 
{
  return (getSharedMembers().deltaT==0);
}

CaConcentrationJunction::~CaConcentrationJunction() 
{
}

