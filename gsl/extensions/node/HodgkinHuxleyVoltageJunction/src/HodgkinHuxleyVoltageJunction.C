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
#include "HodgkinHuxleyVoltageJunction.h"
#include "CG_HodgkinHuxleyVoltageJunction.h"
#include "rndm.h"
#include "GridLayerDescriptor.h"
#define DISTANCE_SQUARED(a,b) ((((a).x-(b).x)*((a).x-(b).x))+(((a).y-(b).y)*((a).y-(b).y))+(((a).z-(b).z)*((a).z-(b).z)))
//#define DEBUG_HH

#ifdef DEBUG_HH
#include "../../../../../nti/SegmentDescriptor.h"
#endif

void HodgkinHuxleyVoltageJunction::initializeJunction(RNG& rng) 
{
  assert(Vnew.size()==1);
  Vcur=Vnew[0];
  assert(dimensions.size()==1);
  DimensionStruct* dimension=dimensions[0];
  if (dimensionInputs.size()==0) area = 1.3333333*M_PI*dimension->r*dimension->r*dimension->r;
  else {
    Array<DimensionStruct*>::iterator iter=dimensionInputs.begin(), end=dimensionInputs.end(); 
    for (; iter!=end; ++iter) {
      
      float R = 0.5 * (0.5 * ( (*iter)->r + dimension->r ) + dimension->r);
      float L = 0.5 * sqrt(DISTANCE_SQUARED(**iter, *dimension));
      
      area += 2.0 * M_PI * R * L;
      //std::cerr<<"area:"<<area<<std::endl<<std::endl;
    }
  }
  float Poar = M_PI/(area * getSharedMembers().Ra);
  Array<DimensionStruct*>::iterator diter=dimensionInputs.begin(), dend=dimensionInputs.end(); 
  for (; diter!=dend; ++diter) {
    float Rb = 0.5 * ( (*diter)->r + dimension->r );
    gAxial.push_back(Poar * Rb * Rb / sqrt(DISTANCE_SQUARED(**diter, *dimension)));
    //std::cerr<<"gAxial:"<<(Poar * Rb * Rb / sqrt(DISTANCE_SQUARED(**diter, *dimension)))<<std::endl<<std::endl;
  }
  if (getSharedMembers().deltaT) {
    cmt = 2.0 * Cm / *(getSharedMembers().deltaT);
  }
#ifdef DEBUG_HH
  std::cerr<<"JUNCTION ("<<dimension->x<<","<<dimension->y<<","<<dimension->z<<","<<dimension->r<<")"<<std::endl;
#endif
}

void HodgkinHuxleyVoltageJunction::predictJunction(RNG& rng) 
{
  assert(cmt>0);
  float conductance = cmt;
  float current = cmt * Vcur;

  conductance += gLeak;
  current += gLeak * getSharedMembers().E_leak;

  Array<ChannelCurrents>::iterator citer=channelCurrents.begin();
  Array<ChannelCurrents>::iterator cend=channelCurrents.end();
  for (; citer!=cend; ++citer) {
    float gloc = (*(citer->conductances))[0];
    conductance += gloc;
    current += gloc * (*(citer->reversalPotentials))[0];
  }
  Array<float*>::iterator iter=receptorReversalPotentials.begin();
  Array<float*>::iterator end=receptorReversalPotentials.end();
  Array<float*>::iterator giter=receptorConductances.begin();
  for (; iter!=end; ++iter, ++giter) {
    conductance += **giter;
    current += **iter * **giter;
  }

  iter=injectedCurrents.begin();
  end=injectedCurrents.end();
  for (; iter!=end; ++iter) {
    current += **iter/area;
  }

  Array<float>::iterator xiter=gAxial.begin(), xend=gAxial.end(); 
  Array<float*>::iterator viter=voltageInputs.begin();
  for (; xiter!=xend; ++xiter, ++viter) {
    current += (*xiter) * ((**viter) - Vcur);
  }

  Vnew[0] = current/conductance;

#ifdef DEBUG_HH
  std::cerr<<getSimulation().getIteration() * *getSharedMembers().deltaT
	   <<" JUNCTION PREDICT"
	   <<" ["<<getSimulation().getRank()<<","<<getNodeIndex()<<","
	   <<getIndex()<<"] "
	   <<"("<<segmentDescriptor.getNeuronIndex(branchData->key)
	   <<","<<segmentDescriptor.getBranchIndex(branchData->key)
	   <<","<<segmentDescriptor.getBranchOrder(branchData->key)
	   <<") {"
	   <<dimension->x<<","<<dimension->y<<","<<dimension->z<<","<<dimension->r<<"} "
	   <<Vnew[0]<<std::endl;
#endif
}

void HodgkinHuxleyVoltageJunction::correctJunction(RNG& rng) 
{
  float conductance = cmt;
  float current = cmt * Vcur;

  conductance += gLeak;
  current += gLeak * getSharedMembers().E_leak;

  Array<ChannelCurrents>::iterator citer=channelCurrents.begin();
  Array<ChannelCurrents>::iterator cend=channelCurrents.end();

  for (; citer!=cend; ++citer) {
    float gloc = (*(citer->conductances))[0];
    conductance += gloc;
    current += gloc * (*(citer->reversalPotentials))[0];
  }

  Array<float*>::iterator iter=receptorReversalPotentials.begin();
  Array<float*>::iterator end=receptorReversalPotentials.end();
  Array<float*>::iterator giter=receptorConductances.begin();
  for (; iter!=end; ++iter, ++giter) {
    conductance += **giter;
    current += **iter * **giter;
  }
  
  iter=injectedCurrents.begin();
  end=injectedCurrents.end();
  for (; iter!=end; ++iter) {
    current += **iter/area;
  }

  Array<float>::iterator xiter=gAxial.begin(), xend=gAxial.end(); 
  Array<float*>::iterator viter=voltageInputs.begin();
  for (; xiter!=xend; ++xiter, ++viter) {
    current += (*xiter) * (**viter);
    conductance += (*xiter);
  }  

  Vnew[0] = current/conductance;

  // This is the swap phase
  Vcur = Vnew[0] = 2.0 * Vnew[0] - Vcur;

#ifdef DEBUG_HH
  std::cerr<<getSimulation().getIteration() * *getSharedMembers().deltaT
	   <<" JUNCTION CORRECT"
	   <<" ["<<getSimulation().getRank()<<","<<getNodeIndex()<<","
	   <<getIndex()<<"] "
	   <<"("<<segmentDescriptor.getNeuronIndex(branchData->key)
	   <<","<<segmentDescriptor.getBranchIndex(branchData->key)
	   <<","<<segmentDescriptor.getBranchOrder(branchData->key)
	   <<") {"
	   <<dimension->x<<","<<dimension->y<<","<<dimension->z<<","<<dimension->r<<"} "
	   <<Vnew[0]<<std::endl;

  Array<DimensionStruct*>::iterator diter=dimensionInputs.begin();
  Array<float*>::iterator vend=voltageInputs.end();
  int c=0;

  for (viter=voltageInputs.begin(); viter!=vend; ++viter, ++diter) {
    std::cerr<<getSimulation().getIteration() * *getSharedMembers().deltaT
               <<" JCT_INPUT_"<<c++
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

bool HodgkinHuxleyVoltageJunction::checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset) 
{
  assert(dimensions.size()==1);
  DimensionStruct* dimension=dimensions[0];
  TissueSite& site=CG_inAttrPset->site;
  bool rval=(site.r==0);
  if (!rval)
    rval=( (site.r*site.r) >= DISTANCE_SQUARED(site, *dimension) );
  return rval;
}

bool HodgkinHuxleyVoltageJunction::confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset) 
{
  return (getSharedMembers().deltaT==0);
}

HodgkinHuxleyVoltageJunction::~HodgkinHuxleyVoltageJunction() 
{
}

