// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "IP3ConcentrationJunction.h"
#include "CG_IP3ConcentrationJunction.h"
#include "rndm.h"
#include "GridLayerDescriptor.h"
#include "MaxComputeOrder.h"
#include "Branch.h"
#include "GlobalNTSConfig.h"

//#define DEBUG_HH
#include <iomanip>
#include <cmath>
#include <cfloat>
#include "SegmentDescriptor.h"
#include "Branch.h"

#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

SegmentDescriptor IP3ConcentrationJunction::_segmentDescriptor;

// NOTE: value = 1e6/(zIP3*Farad)
// zIP3 = valence of IP3 
// Farad = Faraday's constant
#define uM_um_cubed_per_pA_msec 5.18213484752067


#if IP3_CYTO_DYNAMICS == FAST_BUFFERING
#define D_IP3 (getSharedMembers().D_IP3eff)
#else
#define D_IP3 (getSharedMembers().D_IP3)
#endif

// Get cytoplasmic surface area at the compartment i-th 
dyn_var_t IP3ConcentrationJunction::getArea() // Tuan: check ok
{
  dyn_var_t area= 0.0;
#if defined (USE_SOMA_AS_POINT)
  area = 1.0 * FRACTION_SURFACEAREA_CYTO; // [um^2]
#else
  area = dimensions[0]->surface_area * FRACTION_SURFACEAREA_CYTO;
#endif
	return area;
}

// Get cytoplasmic volume at the compartment i-th 
dyn_var_t IP3ConcentrationJunction::getVolume() // Tuan: check ok
{
  dyn_var_t volume = 0.0;
#if defined (USE_SOMA_AS_POINT)
  volume = 1.0 * FRACTIONVOLUME_CYTO; // [um^3]
#else
  volume = dimensions[0]->volume * FRACTIONVOLUME_CYTO;
#endif
  return volume;
}

void IP3ConcentrationJunction::initializeJunction(RNG& rng)
{// explicit junction (which can be soma (with branches are axon/dendrite
  // trees)
  // or a cut point junction 
  // or a branching point junction with 3 or more branches (one from main, 2+ for children
  // branches))
  assert(IP3_new.size() == 1);
  assert(dimensions.size() == 1);

  IP3_cur = IP3_new[0];
  // So, one explicit junction is composed of one compartment 
  // which can be explicit cut-point junction or
  //              explicit branching-point junction
  DimensionStruct* dimension = dimensions[0];  

  volume = getVolume();

  float Pdov = M_PI * D_IP3 / volume;
#define THRESHOLD_SIZE_R_SOMA 2.0 // um (micrometer)
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA &&
      dimension->r > THRESHOLD_SIZE_R_SOMA // to avoid the confusing of spine head
     )//TUAN TODO: consider fixing this
  {
    //for soma: due to large volume, we scale up the [IP3]
    // shell volume = 4/3 * pi * (rsoma^3 - (rsoma-d)^3)
    // with d = shell depth
    // RATIO = somaVolume / shellVolume;
    // currentToConc = getArea() * uM_um_cubed_per_pA_msec / volume * RATIO ;
    // TUAN TODO - 
    dyn_var_t d = 1.0; //[um] - shell depth (default) 
    //dyn_var_t d = 0.5; //[um]  
    //dyn_var_t d = 0.2; //[um]  
    if (GlobalNTS::shellDepth > 0.0)
      d = GlobalNTS::shellDepth;
    dyn_var_t shellVolume = 4.0 / 3.0 * M_PI * 
      (pow(dimension->r,3) - pow(dimension->r - d, 3)) * FRACTIONVOLUME_CYTO;
    currentToConc = getArea() * uM_um_cubed_per_pA_msec / shellVolume;
    //std::cerr << "Cyto total vol: " << volume << "; shell volume: " << shellVolume << std::endl;

    Pdov = M_PI * D_IP3 / shellVolume;

  }
  else
    currentToConc = getArea() * uM_um_cubed_per_pA_msec / volume;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin(),
    dend = dimensionInputs.end();
  for (; diter != dend; ++diter)
  {
    //NOTE: if the junction is the SOMA, we should not use the radius of the SOMA
    //      in calculating the cross-sectional area
    dyn_var_t Rb;
    dyn_var_t distance;
    if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
    {
      //Rb = ((*diter)->r ) * 1.5;  //scaling factor 1.5 means the bigger interface with soma
      //  NOTE: should be applied for Axon hillock only
      Rb = ((*diter)->r );
      //TEST 
      Rb /= SCALING_NECK_FROM_SOMA;
      //END TEST
#ifdef USE_SOMA_AS_ISOPOTENTIAL
      distance = (*diter)->dist2soma - dimension->r; // SOMA is treated as a point source
#else
      //distance= std::fabs((*diter)->dist2soma + dimension->r );
      distance = (*diter)->dist2soma; //NOTE: The dist2soma of the first compartment stemming
      // from soma is always the distance from the center of soma to the center
      // of that compartment
      //TEST 
      distance += STRETCH_SOMA_WITH;
      //END TEST
#endif
      if (distance <= 0)
	std::cerr << "distance = " << distance << ": " << (*diter)->dist2soma << ","<< dimension->r << std::endl;
      assert(distance > 0);
    }else{
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
      Rb = ((*diter)->r); //the small diameter of the branch means small current pass to it
#else
      Rb = 0.5 * ((*diter)->r + dimension->r);
#endif
      distance= std::fabs((*diter)->dist2soma - dimension->dist2soma);
      assert(distance > 0);
    }
    //fAxial.push_back(Pdov * Rb * Rb /
    //                 sqrt(DISTANCE_SQUARED(**diter, *dimension)));
    //dyn_var_t distance= std::fabs((*diter)->dist2soma - dimension->dist2soma);
    fAxial.push_back(Pdov * Rb * Rb / distance );
  }
#ifdef DEBUG_HH
  std::cerr << "IP3_JUNCTION (" << dimension->x << "," << dimension->y << ","
    << dimension->z << "," << dimension->r << ")" << std::endl;
#endif
}

//GOAL: predict IP3_new[0] at offset time (n+1/2) - Crank-Nicolson predictor-corrector scheme
void IP3ConcentrationJunction::predictJunction(RNG& rng)
{
  float LHS = getSharedMembers().bmt;
  float RHS = getSharedMembers().bmt * IP3_cur ;

  Array<ChannelIP3Currents>::iterator citer = channelIP3Currents.begin();
  Array<ChannelIP3Currents>::iterator cend = channelIP3Currents.end();
  for (; citer != cend; ++citer)
  {
    RHS -= currentToConc * (*(citer->currents))[0];
  }

	Array<ChannelIP3Fluxes>::iterator fiter = channelIP3Fluxes.begin();
	Array<ChannelIP3Fluxes>::iterator fend = channelIP3Fluxes.end();
	for (; fiter != fend; fiter++)
	{
		RHS +=  (*fiter->fluxes)[0];
	}

  Array<dyn_var_t*>::iterator iter = receptorIP3Currents.begin();
  Array<dyn_var_t*>::iterator end = receptorIP3Currents.end();
  for (; iter != end; ++iter)
  {
    RHS -= currentToConc * **iter;
  }

	//  3. synapse receptor currents using GHK type equations (gV, gErev)
	//  NOTE: Not available

  //  4. injected currents
  iter = injectedIP3Currents.begin();
  end = injectedIP3Currents.end();
  for (; iter != end; ++iter)
  {
    RHS += **iter * currentToConc / getArea();
  }

#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  Array<dyn_var_t*>::iterator titer = targetReversalIP3Concentration.begin();
  Array<dyn_var_t*>::iterator tend = targetReversalIP3Concentration.end();
  int i = 0;
  for (; titer != tend; ++titer, ++i)
  {
    RHS += *targetInverseTimeIP3Concentration[i] * **titer;
  }
#endif

  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = IP3ConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    RHS += (*xiter) * ((**viter) - IP3_cur);
  }

  IP3_new[0] = RHS / LHS;

#ifdef DEBUG_HH
  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
            << " IP3_JUNCTION PREDICT"
            << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
            << getIndex() << "] "
            << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
            << _segmentDescriptor.getBranchIndex(branchData->key) << ","
            << _segmentDescriptor.getBranchOrder(branchData->key) << ") {"
            << dimensions[0]->x << "," << dimensions[0]->y << ","
            << dimensions[0]->z << ","
            << dimensions[0]->r << ","
            << dimensions[0]->dist2soma << "," << dimensions[0]->surface_area << ","
            << dimensions[0]->volume << "," << dimensions[0]->length << "} " 
            << IP3_new[0] << std::endl;
#endif
}

//GOAL: correct IP3_new[0] at (t+dt/2) 
// and finally update at (t+dt) for IP3_cur, and IP3_new[0]
void IP3ConcentrationJunction::correctJunction(RNG& rng)
{
  float LHS = getSharedMembers().bmt;
  float RHS = getSharedMembers().bmt * IP3_cur;

  Array<ChannelIP3Currents>::iterator citer = channelIP3Currents.begin();
  Array<ChannelIP3Currents>::iterator cend = channelIP3Currents.end();
  for (; citer != cend; ++citer)
  {
    RHS -= currentToConc * (*(citer->currents))[0];
  }

	Array<ChannelIP3Fluxes>::iterator fiter = channelIP3Fluxes.begin();
	Array<ChannelIP3Fluxes>::iterator fend = channelIP3Fluxes.end();
	for (; fiter != fend; fiter++)
	{
		RHS +=  (*fiter->fluxes)[0];
	}

  Array<dyn_var_t*>::iterator iter = receptorIP3Currents.begin();
  Array<dyn_var_t*>::iterator end = receptorIP3Currents.end();
  for (; iter != end; ++iter)
  {
    RHS -= currentToConc * **iter;
  }

  iter = injectedIP3Currents.begin();
  end = injectedIP3Currents.end();
  for (; iter != end; ++iter)
  {
    RHS += **iter * currentToConc / getArea();
  }

  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = IP3ConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    LHS += (*xiter);
    RHS += (*xiter) * (**viter);
  }

#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  Array<dyn_var_t*>::iterator titer = targetReversalIP3Concentration.begin();
  Array<dyn_var_t*>::iterator tend = targetReversalIP3Concentration.end();
  Array<dyn_var_t*>::iterator tviter = targetInverseTimeIP3Concentration.begin();
  for (; titer != tend; ++titer, ++tviter)
  {
    RHS += **tviter * **titer;
    LHS += **tviter ;
  }
#endif

  IP3_new[0] = RHS / LHS;

  // This is the swap phase
  IP3_cur = IP3_new[0] = 2.0 * IP3_new[0] - IP3_cur;

#ifdef DEBUG_HH
	printDebugHH();
#endif
}

void IP3ConcentrationJunction::printDebugHH(std::string phase)
{
	std::cerr << "step,time|" << phase << " [rank,nodeIdx,instanceIdx] " <<
		"(neuronIdx,branchIdx,brchOrder){x,y,z,r | dist2soma,surfarea,volume,len} Vm" << std::endl;
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  std::cerr << getSimulation().getIteration() << "," 
            << getSimulation().getIteration() * *getSharedMembers().deltaT
            << "| " << phase
            << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
            << getIndex() << "] "
            << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
            << _segmentDescriptor.getBranchIndex(branchData->key) << ","
            << _segmentDescriptor.getBranchOrder(branchData->key) << ") {"
            << dimensions[0]->x << "," 
            << dimensions[0]->y << ","
            << dimensions[0]->z << "," 
            << dimensions[0]->r << " | " 
            << dimensions[0]->dist2soma << "," << dimensions[0]->surface_area << ","
            << dimensions[0]->volume << "," << dimensions[0]->length << "} " 
            << IP3_new[0] << std::endl;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin();
  Array<dyn_var_t*>::iterator vend = IP3ConcentrationInputs.end();
  int c = -1;

	std::cerr << "JCT_INPUT_i " <<
		"(neuronIdx,branchIdx,brchOrder, brType, COMPUTEORDER){x,y,z,r | dist2soma,surfarea,volume,len} Vm" << std::endl;
  Array<dyn_var_t*>::iterator viter = IP3ConcentrationInputs.begin();
  for (viter = IP3ConcentrationInputs.begin(); viter != vend; ++viter, ++diter)
  {
    c++;
    std::cerr << " JCT_INPUT_" << c 
      << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
      << std::setw(2) << _segmentDescriptor.getBranchIndex(branchData->key) << ","
      << _segmentDescriptor.getBranchOrder(branchData->key) << ","
      << _segmentDescriptor.getBranchType(branchDataInputs[c]->key) << ","
      << _segmentDescriptor.getComputeOrder(branchData->key) << ") {"
      << std::setprecision(3) << (*diter)->x << "," 
      << std::setprecision(3) << (*diter)->y << "," 
      << std::setprecision(3) << (*diter)->z << ","
      << std::setprecision(3) << (*diter)->r << " | " 
      << (*diter)->dist2soma << "," << (*diter)->surface_area << "," 
      << (*diter)->volume << "," << (*diter)->length  << "} "
      << " " << *(*viter) << std::endl;
  }
}

//TUAN: TODO challenge
//   how to check for 2 sites overlapping
//   if we don't retain the dimension's (x,y,z) coordinate
//  Even if we retain (x,y,z) this value change with the #capsule per compartment
//   and geometric sampling --> so not a good choice
bool IP3ConcentrationJunction::checkSite(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_IP3ConcentrationJunctionInAttrPSet* CG_inAttrPset,
    CG_IP3ConcentrationJunctionOutAttrPSet* CG_outAttrPset)
{
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  TissueSite& site = CG_inAttrPset->site;
  bool rval = (site.r == 0);
  if (!rval) rval = ((site.r * site.r) >= DISTANCE_SQUARED(site, *dimension));
  return rval;
}

bool IP3ConcentrationJunction::confirmUniqueDeltaT(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_IP3ConcentrationJunctionInAttrPSet* CG_inAttrPset,
    CG_IP3ConcentrationJunctionOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}


IP3ConcentrationJunction::~IP3ConcentrationJunction() {}
