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
#include "HodgkinHuxleyVoltage.h"
#include "CG_HodgkinHuxleyVoltage.h"
#include "rndm.h"
#include "GridLayerDescriptor.h"

#define DISTANCE_SQUARED(a,b) ((((a)->x-(b)->x)*((a)->x-(b)->x))+(((a)->y-(b)->y)*((a)->y-(b)->y))+(((a)->z-(b)->z)*((a)->z-(b)->z)))

#define isProximalCase0 (proximalDimension==0) // no flux boundary condition
#define isProximalCase1 (proximalJunction==0 && proximalDimension!=0) // connected to proximal cut or branch point for implicit solve
#define isProximalCase2 (proximalJunction) // connected to proximal junction

#define isDistalCase0 (distalDimensions.size()==0) // no flux boundary condition
#define isDistalCase1 (distalAiis.size()==1) // connected to distal cut point for implicit solve
#define isDistalCase2 (distalAiis.size()==0 && distalInputs.size()==1) // connected to distal junction
#define isDistalCase3 (distalAiis.size()>1) // connected to distal branch point for implicit solve

//#define DEBUG_HH

float HodgkinHuxleyVoltage::getLambda(DimensionStruct* a, DimensionStruct* b) {
  float radius = 0.5*(a->r + b->r);
  float length = DISTANCE_SQUARED(a, b);
  return(radius*radius/(2.0*getSharedMembers().Ra*length*b->r)); /* needs fixing */
}

float HodgkinHuxleyVoltage::getArea(DimensionStruct* a, DimensionStruct* b) {
  float radius = 0.5*(b->r + 0.5*(a->r + b->r));
  float length = 0.5*sqrt(DISTANCE_SQUARED(a, b));
  return(2.0*M_PI*radius*length);
}

float HodgkinHuxleyVoltage::getAij(DimensionStruct* a, DimensionStruct* b, float A) {
  float Rb = 0.5 * ( a->r + b->r );
  return(M_PI*Rb*Rb/(A*getSharedMembers().Ra*sqrt(DISTANCE_SQUARED(a, b))));
}

float HodgkinHuxleyVoltage::getArea(int i) {
  assert(i>=0 && i<branchData->size);
  float area=0.0;
  if (i == branchData->size-1) {
    if (proximalDimension) area += getArea(proximalDimension, dimensions[i]);
  } else {
    area += getArea(dimensions[i+1], dimensions[i]);
  }
  if (i == 0) {
    for (int n = 0; n < distalDimensions.size(); n++) {
      area += getArea(distalDimensions[n], dimensions[i]);
    }
  } else {
    area += getArea(dimensions[i-1], dimensions[i]);
  }
  return area;
}

void HodgkinHuxleyVoltage::initializeVoltage(RNG& rng)
{
  cmt = 2.0 * Cm / *(getSharedMembers().deltaT);

  unsigned size=branchData->size;
  SegmentDescriptor segmentDescriptor;
  computeOrder=segmentDescriptor.getComputeOrder(branchData->key);
  if (isProximalCase2) assert(computeOrder==0);
  if (isDistalCase2) assert(computeOrder==MAX_COMPUTE_ORDER);
  assert(dimensions.size()==size);
  assert(Vnew.size()==size);
  assert(distalDimensions.size()==distalInputs.size());

  if (Vcur.size()!=size) Vcur.increaseSizeTo(size);

  Vcur[0]=Vnew[0];

  if (Aii.size()!=size) Aii.increaseSizeTo(size);
  if (Aip.size()!=size) Aip.increaseSizeTo(size);
  if (Aim.size()!=size) Aim.increaseSizeTo(size);
  if (RHS.size()!=size) RHS.increaseSizeTo(size);

  for (int i=1; i<size; ++i) {
    Vnew[i]=Vnew[0];
    Vcur[i]=Vcur[0];
  }
  for (int i=0; i<size; ++i) {
    Aii[i] = Aip[i] = Aim[i] = RHS[i] = 0.0;
  }

  Array<InjectedCurrent>::iterator iiter= injectedCurrents.begin();
  Array<InjectedCurrent>::iterator iend = injectedCurrents.end();
  for (; iiter != iend; iiter++) {
    if (iiter->index<branchData->size) iiter->area=getArea(iiter->index);
  }

  float Rp, Lp, Rm, Lm;

  Aim[0] = Aip[size-1] = 0;
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (!isProximalCase0) {
    Aip[size-1] = -getLambda(proximalDimension, dimensions[size-1]);
  }
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (isDistalCase1 || isDistalCase2) {
    Aim[0] = -getLambda(distalDimensions[0], dimensions[0]);
  }
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  for (int i = 1; i < size; i++) {
    Aim[i] = -getLambda(dimensions[i-1], dimensions[i]);
  }
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  for (int i = 0; i < size - 1; i++) {
    Aip[i] = -getLambda(dimensions[i+1], dimensions[i]);
  }

  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (isDistalCase3) {
    // Compute total area of the junction...
    float area = getArea(0);

    // Compute Aij[n] for the junction...one of which goes in Aip[0]...
    if (size == 1) {
      Aip[0] = -getAij(proximalDimension, dimensions[0], area);
    } else {
      Aip[0] = -getAij(dimensions[1], dimensions[0], area);
    }
    for (int n = 0; n < distalDimensions.size(); n++) {
      Aij.push_back(-getAij(distalDimensions[n], dimensions[0], area));
    }
  }
}

void HodgkinHuxleyVoltage::doForwardSolve()
{ 
  unsigned size=branchData->size;
 // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  for (int i = 0; i < size; i++) {
    Aii[i] = cmt - Aim[i] - Aip[i] + gLeak;
    RHS[i] = cmt * Vcur[i] + gLeak * getSharedMembers().E_leak;
    /* * * Sum Currents * * */
    Array<ChannelCurrents>::iterator iter= channelCurrents.begin();
    Array<ChannelCurrents>::iterator end = channelCurrents.end();
    for (int k=0; iter != end; iter++, ++k) {
      ShallowArray<float>* conductances = iter->conductances;
      RHS[i] += (*conductances)[i] * (*(iter->reversalPotentials))[(iter->reversalPotentials->size()==1) ? 0:i];
      Aii[i] += (*conductances)[i];
    }
  }
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (isDistalCase3) {
    Aii[0] = cmt - Aip[0] + gLeak;
    RHS[0] = cmt*Vcur[0] + gLeak * getSharedMembers().E_leak;
    for (int n = 0; n < distalInputs.size(); n++) {
      Aii[0] -= Aij[n];
    }
    /* * * Sum Currents * * */
    Array<ChannelCurrents>::iterator citer= channelCurrents.begin();
    Array<ChannelCurrents>::iterator cend = channelCurrents.end();
    for (; citer != cend; citer++) {
      ShallowArray<float>* conductances = citer->conductances;
      RHS[0] += (*conductances)[0] * (*(citer->reversalPotentials))[0];
      Aii[0] += (*conductances)[0];
    }
  }

  Array<ReceptorCurrent>::iterator riter= receptorCurrents.begin();
  Array<ReceptorCurrent>::iterator rend = receptorCurrents.end();
  for (; riter != rend; riter++) {
    int i=riter->index;
    RHS[i] += *(riter->conductance) * *(riter->reversalPotential);
    Aii[i] += *(riter->conductance);
  }

  /* Note: Injected currents comprise current interfaces produced by two different
     categories of models: 1) experimentally injected currents, as in a patch clamp
     electrode in current clamp mode, and 2) electrical synapse currents, as in the 
     current injected from one compartment to another via a gap junction. 

     Since we think of injected currents as positive quantities with units of pA,
     the sign on injected currents is reversed, and the units are pA and not pA/um^2,
     even for electrical synapses.   
  */ 

  Array<InjectedCurrent>::iterator iiter= injectedCurrents.begin();
  Array<InjectedCurrent>::iterator iend = injectedCurrents.end();
  for (; iiter != iend; iiter++) {
   if (iiter->index<branchData->size) RHS[iiter->index] += *(iiter->current) / iiter->area;
  }

 /* * *  Forward Solve Ax = B * * */
  if (isDistalCase1) {
    Aii[0] -= Aim[0]* *distalAips[0]/ *distalAiis[0];
    RHS[0] -= Aim[0]* *distalInputs[0]/ *distalAiis[0];
  } else if (isDistalCase2) {
	// Why do we not adjust Aii[0]? Check.
    RHS[0] -= Aim[0] * *distalInputs[0];
  } else if (isDistalCase3) {
    for (int n = 0; n < distalInputs.size(); n++) {
      Aii[0] -= Aij[n]* *distalAips[n]/ *distalAiis[n];
      RHS[0] -= Aij[n]* *distalInputs[n]/ *distalAiis[n];
    }
  }
  for (int i = 1; i < size; i++) {
    Aii[i] -= Aim[i]*Aip[i-1]/Aii[i-1];
    RHS[i] -= Aim[i]*RHS[i-1]/Aii[i-1];
  }
}

void HodgkinHuxleyVoltage::doBackwardSolve()
{
  unsigned size=branchData->size;
  if (isProximalCase0) {
    Vnew[size-1] = RHS[size-1]/Aii[size-1];
  } else {
    Vnew[size-1] = (RHS[size-1] - Aip[size-1] * *proximalVoltage)/Aii[size-1];
  }
  for (int i = size-2; i >= 0; i--) {
    Vnew[i] = (RHS[i] - Aip[i]*Vnew[i+1])/Aii[i];
  }
}

void HodgkinHuxleyVoltage::solve(RNG& rng)
{
  if (computeOrder==0) {
    doForwardSolve();
    doBackwardSolve();
  }
}

#if MAX_COMPUTE_ORDER>0
void HodgkinHuxleyVoltage::forwardSolve1(RNG& rng)
{
  if (computeOrder==1) {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve1(RNG& rng)
{
  if (computeOrder==1) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>1
void HodgkinHuxleyVoltage::forwardSolve2(RNG& rng)
{
  if (computeOrder==2) {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve2(RNG& rng)
{
  if (computeOrder==2) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>2
void HodgkinHuxleyVoltage::forwardSolve3(RNG& rng)
{
  if (computeOrder==3) {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve3(RNG& rng)
{
  if (computeOrder==3) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>3
void HodgkinHuxleyVoltage::forwardSolve4(RNG& rng)
{
  if (computeOrder==4) {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve4(RNG& rng)
{
  if (computeOrder==4) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>4
void HodgkinHuxleyVoltage::forwardSolve5(RNG& rng)
{
  if (computeOrder==5) {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve5(RNG& rng)
{
  if (computeOrder==5) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>5
void HodgkinHuxleyVoltage::forwardSolve6(RNG& rng)
{
  if (computeOrder==6) {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve6(RNG& rng)
{
  if (computeOrder==6) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>6
void HodgkinHuxleyVoltage::forwardSolve7(RNG& rng)
{
  if (computeOrder==7) {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve7(RNG& rng)
{
  if (computeOrder==7) doBackwardSolve();
}
#endif

void HodgkinHuxleyVoltage::finish(RNG& rng)
{
  unsigned size=branchData->size;
#ifdef DEBUG_HH
  SegmentDescriptor segmentDescriptor;
  for (int i=0; i<size; ++i) {
    std::cerr<<float(getSimulation().getIteration()) * *getSharedMembers().deltaT
	     <<" BRANCH"
	     <<" ["<<getSimulation().getRank()<<","<<getNodeIndex()<<","
	     <<getIndex()<<","<<i<<"] "
	     <<"("<<segmentDescriptor.getNeuronIndex(branchData->key)
	     <<","<<segmentDescriptor.getBranchIndex(branchData->key)
	     <<","<<segmentDescriptor.getBranchOrder(branchData->key)
	     <<") |"<<isDistalCase0<<"|"<<isDistalCase1<<"|"<<isDistalCase2<<"|"<<isDistalCase3<<"|"
	     <<isProximalCase0<<"|"<<isProximalCase1<<"|"<<isProximalCase2<<"|"<<" {"
	     <<dimensions[i].x<<","<<dimensions[i].y<<","<<dimensions[i].z<<","<<dimensions[i].r<<"} "
	     <<Vnew[i]<<" "<<std::endl;
  }
#endif
  for (int i=0; i<size; ++i) {
    Vcur[i] = Vnew[i] = 2.0 * Vnew[i] - Vcur[i];
  }
}

void HodgkinHuxleyVoltage::setReceptorCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
  assert(receptorCurrents.size()>0);
  receptorCurrents[receptorCurrents.size()-1].index=CG_inAttrPset->idx;
}

void HodgkinHuxleyVoltage::setInjectedCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
  assert(injectedCurrents.size()>0);
  TissueSite& site=CG_inAttrPset->site;
  if (site.r!=0) {
    for (int i=0; i<dimensions.size(); ++i) {
      if ( (site.r*site.r) >= DISTANCE_SQUARED(&site, dimensions[i]) ) {
	CurrentProducer* CG_CurrentProducerPtr = dynamic_cast<CurrentProducer*>(CG_variable);
	if (CG_CurrentProducerPtr == 0) {
	  std::cerr << "Dynamic Cast of CurrentProducer failed in HodgkinHuxleyVoltage" << std::endl;
	  exit(-1);
	}
	injectedCurrents.increase();
	injectedCurrents[injectedCurrents.size()-1].current = CG_CurrentProducerPtr->CG_get_CurrentProducer_current();
	injectedCurrents[injectedCurrents.size()-1].index=i;
	checkAndAddPreVariable(CG_variable);
      }
    }
  }
  else if (CG_inAttrPset->idx<0) {
    injectedCurrents[injectedCurrents.size()-1].index=0;
    for (int i=1; i<branchData->size; ++i) {
      CurrentProducer* CG_CurrentProducerPtr = dynamic_cast<CurrentProducer*>(CG_variable);
      if (CG_CurrentProducerPtr == 0) {
	std::cerr << "Dynamic Cast of CurrentProducer failed in HodgkinHuxleyVoltage" << std::endl;
	exit(-1);
      }
      injectedCurrents.increase();
      injectedCurrents[injectedCurrents.size()-1].current = CG_CurrentProducerPtr->CG_get_CurrentProducer_current();
      injectedCurrents[injectedCurrents.size()-1].index=i;
      checkAndAddPreVariable(CG_variable);
    }
  }
  else {
    injectedCurrents[injectedCurrents.size()-1].index=CG_inAttrPset->idx;
  }
}

void HodgkinHuxleyVoltage::setProximalJunction(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
  proximalJunction=true;
}

bool HodgkinHuxleyVoltage::checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset) 
{
  TissueSite& site=CG_inAttrPset->site;
  bool atSite=(site.r==0);
  for (int i=0; !atSite && i<dimensions.size(); ++i)
    atSite=( (site.r*site.r) >= DISTANCE_SQUARED(&site, dimensions[i]) );
  return atSite;
}

bool HodgkinHuxleyVoltage::confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT==0);
}

HodgkinHuxleyVoltage::~HodgkinHuxleyVoltage()
{
}
