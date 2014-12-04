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
#include "CaConcentration.h"
#include "CG_CaConcentration.h"
#include "rndm.h"
#include "GridLayerDescriptor.h"
#define DISTANCE_SQUARED(a,b) ((((a)->x-(b)->x)*((a)->x-(b)->x))+(((a)->y-(b)->y)*((a)->y-(b)->y))+(((a)->z-(b)->z)*((a)->z-(b)->z)))
#define uM_um_cubed_per_pA_msec 5.18213484752067

#define isProximalCase0 (proximalDimension==0) // no flux boundary condition
#define isProximalCase1 (proximalJunction==0 && proximalDimension!=0) // connected to proximal cut or branch point for implicit solve
#define isProximalCase2 (proximalJunction) // connected to proximal junction

#define isDistalCase0 (distalDimensions.size()==0) // no flux boundary condition
#define isDistalCase1 (distalAiis.size()==1) // connected to distal cut point for implicit solve
#define isDistalCase2 (distalAiis.size()==0 && distalInputs.size()==1) // connected to distal junction
#define isDistalCase3 (distalAiis.size()>1) // connected to distal branch point for implicit solve

//#define DEBUG_HH

float CaConcentration::getLambda(DimensionStruct* a, DimensionStruct* b) {
  float radius = 0.5*(a->r + b->r);
  float length = DISTANCE_SQUARED(a, b);
  return(getSharedMembers().DCa*radius*radius/(length*b->r*b->r)); /* needs fixing */
}

/* FIX */
float CaConcentration::getAij(DimensionStruct* a, DimensionStruct* b, float V) {
  float Rb = 0.5 * ( a->r + b->r );
  return(M_PI*Rb*Rb*getSharedMembers().DCa/(V*sqrt(DISTANCE_SQUARED(a, b))));
}

/* CHECK */
float CaConcentration::getArea(DimensionStruct* a, DimensionStruct* b) {
  float radius = 0.5*(b->r + 0.5*(a->r + b->r));
  float length = 0.5*sqrt(DISTANCE_SQUARED(a, b));
  return(2.0*M_PI*radius*length);
}

/* CHECK */
float CaConcentration::getArea(int i) {
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

float CaConcentration::getVolume(DimensionStruct* a, DimensionStruct* b) {
  float radius = 0.5*(b->r + 0.5*(a->r + b->r));
  float length = 0.5*sqrt(DISTANCE_SQUARED(a, b));
  return(M_PI*radius*radius*length);
}

float CaConcentration::getVolume(int i) {
  assert(i>=0 && i<branchData->size);
  float volume=0.0;
  if (i == branchData->size-1) {
    if (proximalDimension) volume += getVolume(proximalDimension, dimensions[i]);
  } else {
    volume += getVolume(dimensions[i+1], dimensions[i]);
  }
  if (i == 0) {
    for (int n = 0; n < distalDimensions.size(); n++) {
      volume += getVolume(distalDimensions[n], dimensions[i]);
    }
  } else {
    volume += getVolume(dimensions[i-1], dimensions[i]);
  }
  return volume;
}

void CaConcentration::initializeCaConcentration(RNG& rng)
{
  unsigned size=branchData->size;
  SegmentDescriptor segmentDescriptor;
  computeOrder=segmentDescriptor.getComputeOrder(branchData->key);
  if (isProximalCase2) assert(computeOrder==0);
  if (isDistalCase2) assert(computeOrder==MAX_COMPUTE_ORDER);
  assert(Ca_new.size()==size);
  assert(dimensions.size()==size);
  assert(distalDimensions.size()==distalInputs.size());

  if (Ca_cur.size()!=size) Ca_cur.increaseSizeTo(size);

  Ca_cur[0]=Ca_new[0];

  if (Aii.size()!=size) Aii.increaseSizeTo(size);
  if (Aip.size()!=size) Aip.increaseSizeTo(size);
  if (Aim.size()!=size) Aim.increaseSizeTo(size);
  if (RHS.size()!=size) RHS.increaseSizeTo(size);
  if (currentToConc.size()!=size) currentToConc.increaseSizeTo(size);

  for (int i=1; i<size; ++i) {
    Ca_new[i]=Ca_new[0];
    Ca_cur[i]=Ca_cur[0];
  }
  for (int i=0; i<size; ++i) {
    Aii[i] = Aip[i] = Aim[i] = RHS[i] = 0.0;
    currentToConc[i] = getArea(i) * uM_um_cubed_per_pA_msec / getVolume(i);
  }

  Array<InjectedCaCurrent>::iterator iiter = injectedCaCurrents.begin();
  Array<InjectedCaCurrent>::iterator iend = injectedCaCurrents.end();
  for (; iiter != iend; iiter++) {
    if (iiter->index<branchData->size) iiter->currentToConc=uM_um_cubed_per_pA_msec/getVolume(iiter->index);
  }

  float Rp, Lp, Rm, Lm;

  Aim[0] = Aip[size-1] = 0;

  if (!isProximalCase0) {
    Aip[size-1] = -getLambda(proximalDimension, dimensions[size-1]);
  }

  if (isDistalCase1 || isDistalCase2) {
    Aim[0] = -getLambda(distalDimensions[0], dimensions[0]);
  }

  for (int i = 1; i < size; i++) {
    Aim[i] = -getLambda(dimensions[i-1], dimensions[i]);
  }

  for (int i = 0; i < size - 1; i++) {
    Aip[i] = -getLambda(dimensions[i+1], dimensions[i]);
  }

  /* FIX */
  if (isDistalCase3) {
    // Compute total volume of the junction...
    float volume = getVolume(0);

    // Compute Aij[n] for the junction...one of which goes in Aip[0]...
    if (size == 1) {
      Aip[0] = -getAij(proximalDimension, dimensions[0], volume);
    } else {
      Aip[0] = -getAij(dimensions[1], dimensions[0], volume);
    }
    for (int n = 0; n < distalDimensions.size(); n++) {
      Aij.push_back(-getAij(distalDimensions[n], dimensions[0], volume));
    }
  }
}

void CaConcentration::doForwardSolve()
{
  unsigned size=branchData->size;
  for (int i = 0; i < size; i++) {
    Aii[i] = getSharedMembers().bmt - Aim[i] - Aip[i];
    RHS[i] = getSharedMembers().bmt * Ca_cur[i];
    /* * * Sum Currents * * */
    Array<ChannelCaCurrents>::iterator iter= channelCaCurrents.begin();
    Array<ChannelCaCurrents>::iterator end = channelCaCurrents.end();
    for (; iter != end; iter++) {
      RHS[i] -= currentToConc[i] * (*iter->currents)[i];
    }

    /* This is a simple implementation of calcium extrusion. To be elaborated as needed. */
    RHS[i] -= CaClearance * (Ca_cur[i] - getSharedMembers().CaBaseline);
  }

  /* FIX */  
  if (isDistalCase3) {
    Aii[0] = getSharedMembers().bmt - Aip[0];
    RHS[0] = getSharedMembers().bmt * Ca_cur[0];
    for (int n = 0; n < distalInputs.size(); n++) {
      Aii[0] -= Aij[n];
    }
    /* * * Sum Currents * * */
    Array<ChannelCaCurrents>::iterator citer= channelCaCurrents.begin();
    Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
    for (; citer != cend; citer++) {
      RHS[0] -= currentToConc[0] * (*citer->currents)[0];
    }
  }

  Array<ReceptorCaCurrent>::iterator riter= receptorCaCurrents.begin();
  Array<ReceptorCaCurrent>::iterator rend = receptorCaCurrents.end();
  for (; riter != rend; riter++) {
    int i=riter->index;
    RHS[i] -= currentToConc[i] * *(riter->current);
  }

  Array<InjectedCaCurrent>::iterator iiter= injectedCaCurrents.begin();
  Array<InjectedCaCurrent>::iterator iend = injectedCaCurrents.end();
  for (; iiter != iend; iiter++) {
    if (iiter->index<branchData->size) RHS[iiter->index] += *(iiter->current) * iiter->currentToConc;
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

void CaConcentration::doBackwardSolve()
{
  unsigned size=branchData->size;
  if (isProximalCase0) {
    Ca_new[size-1] = RHS[size-1]/Aii[size-1];
  } else {
    Ca_new[size-1] = (RHS[size-1]- Aip[size-1] * *proximalCaConcentration)/Aii[size-1];
  }
  for (int i = size-2; i >= 0; i--) {
    Ca_new[i] = (RHS[i] - Aip[i]*Ca_new[i+1])/Aii[i];
  }
}

void CaConcentration::solve(RNG& rng)
{
  if (computeOrder==0) {
    doForwardSolve();
    doBackwardSolve();
  }
}

#if MAX_COMPUTE_ORDER>0
void CaConcentration::forwardSolve1(RNG& rng)
{
  if (computeOrder==1) {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve1(RNG& rng)
{
  if (computeOrder==1) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>1
void CaConcentration::forwardSolve2(RNG& rng)
{
  if (computeOrder==2) {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve2(RNG& rng)
{
  if (computeOrder==2) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>2
void CaConcentration::forwardSolve3(RNG& rng)
{
  if (computeOrder==3) {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve3(RNG& rng)
{
  if (computeOrder==3) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>3
void CaConcentration::forwardSolve4(RNG& rng)
{
  if (computeOrder==4) {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve4(RNG& rng)
{
  if (computeOrder==4) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>4
void CaConcentration::forwardSolve5(RNG& rng)
{
  if (computeOrder==5) {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve5(RNG& rng)
{
  if (computeOrder==5) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>5
void CaConcentration::forwardSolve6(RNG& rng)
{
  if (computeOrder==6) {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve6(RNG& rng)
{
  if (computeOrder==6) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER>6
void CaConcentration::forwardSolve7(RNG& rng)
{
  if (computeOrder==7) {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve7(RNG& rng)
{
  if (computeOrder==7) doBackwardSolve();
}
#endif

void CaConcentration::finish(RNG& rng)
{
  unsigned size=branchData->size;
#ifdef DEBUG_HH
  SegmentDescriptor segmentDescriptor;
  for (int i=0; i<size; ++i) {
    std::cerr<<float(getSimulation().getIteration()) * *getSharedMembers().deltaT
	     <<" CA_BRANCH"
	     <<" ["<<getSimulation().getRank()<<","<<getNodeIndex()<<","
	     <<getIndex()<<","<<i<<"] "
	     <<"("<<segmentDescriptor.getNeuronIndex(branchData->key)
	     <<","<<segmentDescriptor.getBranchIndex(branchData->key)
	     <<","<<segmentDescriptor.getBranchOrder(branchData->key)
	     <<") |"<<isDistalCase0<<"|"<<isDistalCase1<<"|"<<isDistalCase2<<"|"<<isDistalCase3<<"|"
	     <<isProximalCase0<<"|"<<isProximalCase1<<"|"<<isProximalCase2<<"|"<<" {"
	     <<dimensions[i]->x<<","<<dimensions[i]->y<<","<<dimensions[i]->z<<","<<dimensions[i]->r<<"} "
	     <<Ca_new[i]<<" "<<std::endl;
  }
#endif
  for (int i=0; i<size; ++i) {
    Ca_cur[i] = Ca_new[i] = 2.0 * Ca_new[i] - Ca_cur[i];
    assert(Ca_new[i]>=0);
  }
}

void CaConcentration::setReceptorCaCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
  assert(receptorCaCurrents.size()>0);
  receptorCaCurrents[receptorCaCurrents.size()-1].index=CG_inAttrPset->idx;
}

void CaConcentration::setInjectedCaCurrent(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
  assert(injectedCaCurrents.size()>0);
  TissueSite& site=CG_inAttrPset->site;
  if (site.r!=0) {
    for (int i=0; i<dimensions.size(); ++i) {
      if ( (site.r*site.r) >= DISTANCE_SQUARED(&site, dimensions[i]) ) {
	CaCurrentProducer* CG_CaCurrentProducerPtr = dynamic_cast<CaCurrentProducer*>(CG_variable);
	if (CG_CaCurrentProducerPtr == 0) {
	  std::cerr << "Dynamic Cast of CurrentProducer failed in HodgkinHuxleyVoltage" << std::endl;
	  exit(-1);
	}
	injectedCaCurrents.increase();
	injectedCaCurrents[injectedCaCurrents.size()-1].current = CG_CaCurrentProducerPtr->CG_get_CaCurrentProducer_CaCurrent();
	injectedCaCurrents[injectedCaCurrents.size()-1].index=i;
	checkAndAddPreVariable(CG_variable);
      }
    }
  }
  else if (CG_inAttrPset->idx<0) {
    injectedCaCurrents[injectedCaCurrents.size()-1].index=0;
    for (int i=1; i<branchData->size; ++i) {
      CaCurrentProducer* CG_CaCurrentProducerPtr = dynamic_cast<CaCurrentProducer*>(CG_variable);
      if (CG_CaCurrentProducerPtr == 0) {
	std::cerr << "Dynamic Cast of CurrentProducer failed in HodgkinHuxleyVoltage" << std::endl;
	exit(-1);
      }
      injectedCaCurrents.increase();
      injectedCaCurrents[injectedCaCurrents.size()-1].current = CG_CaCurrentProducerPtr->CG_get_CaCurrentProducer_CaCurrent();
      injectedCaCurrents[injectedCaCurrents.size()-1].index=i;
      checkAndAddPreVariable(CG_variable);
    }
  }
  else {
    injectedCaCurrents[injectedCaCurrents.size()-1].index=CG_inAttrPset->idx;
  }
}

void CaConcentration::setProximalJunction(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
  proximalJunction=true;
}

bool CaConcentration::checkSite(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset) 
{
  TissueSite& site=CG_inAttrPset->site;
  bool atSite=(site.r==0);
  for (int i=0; !atSite && i<dimensions.size(); ++i)
    atSite=( (site.r*site.r) >= DISTANCE_SQUARED(&site, dimensions[i]) );
  return atSite;
}

bool CaConcentration::confirmUniqueDeltaT(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT==0);
}

CaConcentration::~CaConcentration()
{
}

