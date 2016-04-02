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
#include "MaxComputeOrder.h"

//#define DEBUG_HH

#include <cmath>
#include <cfloat>
#include "SegmentDescriptor.h"
#include "Branch.h"

#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))
#define uM_um_cubed_per_pA_msec 5.18213484752067

// Get cytoplasmic surface area at the compartment i-th 
dyn_var_t CaConcentrationJunction::getArea() // Tuan: check ok
{
  dyn_var_t area= 0.0;
  area = dimensions[0]->surface_area * FRACTION_SURFACEAREA_CYTO;
	return area;
}

// Get cytoplasmic volume at the compartment i-th 
dyn_var_t CaConcentrationJunction::getVolume() // Tuan: check ok
{
  dyn_var_t volume = 0.0;
  volume = dimensions[0]->volume * FRACTIONVOLUME_CYTO;
	return volume;
}

void CaConcentrationJunction::initializeJunction(RNG& rng)
{// explicit junction (which can be soma (with branches are axon/dendrite
  // trees)
  // or a cut point junction 
	// or a branching point junction with 3 or more branches (one from main, 2+ for children
  // branches))
  SegmentDescriptor segmentDescriptor;
#ifdef DEBUG_ASSERT
  assert(Ca_new.size() == 1);
  assert(dimensions.size() == 1);
#endif

  Ca_cur = Ca_new[0];
  // So, one explicit junction is composed of one compartment 
  // which can be explicit cut-point junction or
  //              explicit branching-point junction
  DimensionStruct* dimension = dimensions[0];  

  Array<DimensionStruct*>::iterator iter = dimensionInputs.begin(),
                                    end = dimensionInputs.end();

  volume = getVolume();

  float Pdov = M_PI * getSharedMembers().DCa / volume;
  currentToConc = getArea() * uM_um_cubed_per_pA_msec / volume;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin(),
                                    dend = dimensionInputs.end();
  for (; diter != dend; ++diter)
  {
    float Rb = 0.5 * ((*diter)->r + dimension->r);
    //fAxial.push_back(Pdov * Rb * Rb /
    //                 sqrt(DISTANCE_SQUARED(**diter, *dimension)));
	dyn_var_t length= std::fabs((*diter)->dist2soma - dimension->dist2soma);
	fAxial.push_back(Pdov * Rb * Rb / length );
  }
#ifdef DEBUG_HH
  std::cerr << "CA_JUNCTION (" << dimension->x << "," << dimension->y << ","
            << dimension->z << "," << dimension->r << ")" << std::endl;
#endif
}

void CaConcentrationJunction::predictJunction(RNG& rng)
{
#if CALCIUM_CYTO_DYNAMICS == FAST_BUFFERING
  assert(getSharedMembers().bmt > 0);
  float LHS = getSharedMembers().bmt;
  float RHS = getSharedMembers().bmt * Ca_cur ;
#elif CALCIUM_CYTO_DYNAMICS == REGULAR_BUFFERING
		 do something here
#endif


  Array<ChannelCaCurrents>::iterator citer = channelCaCurrents.begin();
  Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
  for (; citer != cend; ++citer)
  {
    RHS -= currentToConc * (*(citer->currents))[0];
  }

	Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
	Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
	for (; fiter != fend; fiter++)
	{
		RHS +=  (*fiter->fluxes)[0];
	}

  Array<dyn_var_t*>::iterator iter = receptorCaCurrents.begin();
  Array<dyn_var_t*>::iterator end = receptorCaCurrents.end();
  for (; iter != end; ++iter)
  {
    RHS -= currentToConc * **iter;
  }

  iter = injectedCaCurrents.begin();
  end = injectedCaCurrents.end();
  for (; iter != end; ++iter)
  {
    RHS += **iter * currentToConc / getArea();
  }

  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    RHS += (*xiter) * ((**viter) - Ca_cur);
  }

  Ca_new[0] = RHS / LHS;

#ifdef DEBUG_HH
  SegmentDescriptor segmentDescriptor;
  DimensionStruct* dimension = dimensions[0];  
  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
            << " CA_JUNCTION PREDICT"
            << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
            << getIndex() << "] "
            << "(" << segmentDescriptor.getNeuronIndex(branchData->key) << ","
            << segmentDescriptor.getBranchIndex(branchData->key) << ","
            << segmentDescriptor.getBranchOrder(branchData->key) << ") {"
            << dimension->x << "," << dimension->y << "," << dimension->z << ","
            << dimension->r << "} " << Ca_new[0] << std::endl;
#endif
}

void CaConcentrationJunction::correctJunction(RNG& rng)
{
#if CALCIUM_CYTO_DYNAMICS == FAST_BUFFERING
  assert(getSharedMembers().bmt > 0);
  float LHS = getSharedMembers().bmt;
  float RHS = getSharedMembers().bmt * Ca_cur;
#elif CALCIUM_CYTO_DYNAMICS == REGULAR_BUFFERING
		 do something here
#endif

  Array<ChannelCaCurrents>::iterator citer = channelCaCurrents.begin();
  Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
  for (; citer != cend; ++citer)
  {
    RHS -= currentToConc * (*(citer->currents))[0];
  }

	Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
	Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
	for (; fiter != fend; fiter++)
	{
		RHS +=  (*fiter->fluxes)[0];
	}

  Array<dyn_var_t*>::iterator iter = receptorCaCurrents.begin();
  Array<dyn_var_t*>::iterator end = receptorCaCurrents.end();
  for (; iter != end; ++iter)
  {
    RHS -= currentToConc * **iter;
  }

  iter = injectedCaCurrents.begin();
  end = injectedCaCurrents.end();
  for (; iter != end; ++iter)
  {
    RHS += **iter * currentToConc / getArea();
  }

  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    LHS += (*xiter);
    RHS += (*xiter) * (**viter);
  }

  Ca_new[0] = RHS / LHS;

  // This is the swap phase
  Ca_cur = Ca_new[0] = 2.0 * Ca_new[0] - Ca_cur;

#ifdef DEBUG_HH
  SegmentDescriptor segmentDescriptor;
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
            << " CA_JUNCTION CORRECT"
            << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
            << getIndex() << "] "
            << "(" << segmentDescriptor.getNeuronIndex(branchData->key) << ","
            << segmentDescriptor.getBranchIndex(branchData->key) << ","
            << segmentDescriptor.getBranchOrder(branchData->key) << ") {"
            << dimension->x << "," << dimension->y << "," << dimension->z << ","
            << dimension->r << "} " << Ca_new[0] << std::endl;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin();
  Array<dyn_var_t*>::iterator vend = CaConcentrationInputs.end();
  int c = 0;

  for (viter = CaConcentrationInputs.begin(); viter != vend; ++viter, ++diter)
  {
    std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
              << " CA_JCT_INPUT_" << c++ << " [" << getSimulation().getRank()
              << "," << getNodeIndex() << "," << getIndex() << "] "
              << "(" << segmentDescriptor.getNeuronIndex(branchData->key) << ","
              << segmentDescriptor.getBranchIndex(branchData->key) << ","
              << segmentDescriptor.getBranchOrder(branchData->key) << ","
              << segmentDescriptor.getComputeOrder(branchData->key) << ") {"
              << (*diter)->x << "," << (*diter)->y << "," << (*diter)->z << ","
              //<< (*diter)->r << "} " << DISTANCE_SQUARED(*(*diter), *dimension)
              << (*diter)->r << "} " << (((*diter))->dist2soma - dimension->dist2soma)
              << " " << *(*viter) << std::endl;
  }
#endif
}

bool CaConcentrationJunction::checkSite(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset,
    CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset)
{
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  TissueSite& site = CG_inAttrPset->site;
  bool rval = (site.r == 0);
  if (!rval) rval = ((site.r * site.r) >= DISTANCE_SQUARED(site, *dimension));
  return rval;
}

bool CaConcentrationJunction::confirmUniqueDeltaT(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset,
    CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}


CaConcentrationJunction::~CaConcentrationJunction() {}
