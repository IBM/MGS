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
#include "MaxComputeOrder.h"

//#define DEBUG_HH
#include "SegmentDescriptor.h"

#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

// Get biomembrane surface area at the compartment i-th 
dyn_var_t HodgkinHuxleyVoltageJunction::getArea() // Tuan: check ok
{
  dyn_var_t area= 0.0;
  area = dimensions[0]->surface_area;
}

void HodgkinHuxleyVoltageJunction::initializeJunction(RNG& rng)
{ // explicit junction (which can be soma (with branches are axon/dendrite
  // trees)
  // or a cut point junction 
	// or a branching point junction with 3 or more branches (one from main, 2+ for children
  // branches))
  SegmentDescriptor segmentDescriptor;
#ifdef DEBUG_ASSERT
  assert(Vnew.size() == 1);
  assert(dimensions.size() == 1);
#endif

  Vcur = Vnew[0];
  // So, one explicit junction is composed of one compartment 
  // which can be explicit cut-point junction or
  //              explicit branching-point junction
  DimensionStruct* dimension = dimensions[0];  
                                             
  area = getArea();
#ifdef DEBUG_HH
  // check 'bouton' neuron ???
  if (segmentDescriptor.getNeuronIndex(branchData->key) == 2)
  {
    printf(" --> Area = %lf\n", area);
    // std::cerr << "area: " << area << std::endl;
  }
#endif

  dyn_var_t Poar = M_PI / (area * getSharedMembers().Ra);  // Pi-over-(area *
                                                           // axial-resistance)
  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin(),
                                    dend = dimensionInputs.end();
  for (; diter != dend; ++diter)
  {
    dyn_var_t Rb = 0.5 * ((*diter)->r + dimension->r);
    dyn_var_t distance = (*diter)->dist2soma - dimension->dist2soma;
    gAxial.push_back(Poar * Rb * Rb / distance);
  }
  if (getSharedMembers().deltaT)
  {
    cmt = 2.0 * Cm / *(getSharedMembers().deltaT);
  }
#ifdef DEBUG_HH
  std::cerr << "JUNCTION (" << dimension->x << "," << dimension->y << ","
            << dimension->z << "," << dimension->r << ")" << std::endl;
#endif
}

void HodgkinHuxleyVoltageJunction::predictJunction(RNG& rng)
{
  assert(cmt > 0);
  dyn_var_t conductance = cmt;
  dyn_var_t current = cmt * Vcur;

  conductance += gLeak;
  current += gLeak * getSharedMembers().E_leak;

  Array<ChannelCurrents>::iterator citer = channelCurrents.begin();
  Array<ChannelCurrents>::iterator cend = channelCurrents.end();
  for (; citer != cend; ++citer)
  {
    dyn_var_t gloc = (*(citer->conductances))[0];
    conductance += gloc;
    current += gloc * (*(citer->reversalPotentials))[0];
  }
  Array<dyn_var_t*>::iterator iter = receptorReversalPotentials.begin();
  Array<dyn_var_t*>::iterator end = receptorReversalPotentials.end();
  Array<dyn_var_t*>::iterator giter = receptorConductances.begin();
  for (; iter != end; ++iter, ++giter)
  {
    conductance += **giter;
    current += **iter * **giter;
  }

  iter = injectedCurrents.begin();
  end = injectedCurrents.end();
  for (; iter != end; ++iter)
  {
    current += **iter / area;
  }

  Array<dyn_var_t>::iterator xiter = gAxial.begin(), xend = gAxial.end();
  Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    current += (*xiter) * ((**viter) - Vcur);
  }

  Vnew[0] = current / conductance;

#ifdef DEBUG_HH
  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
            << " JUNCTION PREDICT"
            << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
            << getIndex() << "] "
            << "(" << segmentDescriptor.getNeuronIndex(branchData->key) << ","
            << segmentDescriptor.getBranchIndex(branchData->key) << ","
            << segmentDescriptor.getBranchOrder(branchData->key) << ") {"
            << dimensions[0]->x << "," << dimensions[0]->y << ","
            << dimensions[0]->z << "," 
						<< dimensions[0]->r << "} " << Vnew[0]
            << std::endl;
#endif
}

void HodgkinHuxleyVoltageJunction::correctJunction(RNG& rng)
{
  dyn_var_t conductance = cmt;
  dyn_var_t current = cmt * Vcur;

  conductance += gLeak;
  current += gLeak * getSharedMembers().E_leak;

  Array<ChannelCurrents>::iterator citer = channelCurrents.begin();
  Array<ChannelCurrents>::iterator cend = channelCurrents.end();

  for (; citer != cend; ++citer)
  {
    dyn_var_t gloc = (*(citer->conductances))[0];
    conductance += gloc;
    current += gloc * (*(citer->reversalPotentials))[0];
  }

  Array<dyn_var_t*>::iterator iter = receptorReversalPotentials.begin();
  Array<dyn_var_t*>::iterator end = receptorReversalPotentials.end();
  Array<dyn_var_t*>::iterator giter = receptorConductances.begin();
  for (; iter != end; ++iter, ++giter)
  {
    conductance += **giter;
    current += **iter * **giter;
  }

  iter = injectedCurrents.begin();
  end = injectedCurrents.end();
  for (; iter != end; ++iter)
  {
    current += **iter / area;
  }

  Array<dyn_var_t>::iterator xiter = gAxial.begin(), xend = gAxial.end();
  Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    current += (*xiter) * (**viter);
    conductance += (*xiter);
  }

  Vnew[0] = current / conductance;

  // This is the swap phase
  Vcur = Vnew[0] = 2.0 * Vnew[0] - Vcur;

#ifdef DEBUG_HH
  SegmentDescriptor segmentDescriptor;
  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
            << " JUNCTION CORRECT"
            << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
            << getIndex() << "] "
            << "(" << segmentDescriptor.getNeuronIndex(branchData->key) << ","
            << segmentDescriptor.getBranchIndex(branchData->key) << ","
            << segmentDescriptor.getBranchOrder(branchData->key) << ") {"
            << dimensions[0]->x << "," << dimensions[0]->y << ","
            << dimensions[0]->z << "," 
						<< dimensions[0]->r << "} " << Vnew[0]
            << std::endl;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin();
  Array<dyn_var_t*>::iterator vend = voltageInputs.end();
  int c = 0;

  for (viter = voltageInputs.begin(); viter != vend; ++viter, ++diter)
  {
    std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
              << " JCT_INPUT_" << c++ << " [" << getSimulation().getRank()
              << "," << getNodeIndex() << "," << getIndex() << "] "
              << "(" << segmentDescriptor.getNeuronIndex(branchData->key) << ","
              << segmentDescriptor.getBranchIndex(branchData->key) << ","
              << segmentDescriptor.getBranchOrder(branchData->key) << ","
              << segmentDescriptor.getComputeOrder(branchData->key) << ") {"
              << (*diter)->x << "," << (*diter)->y << "," << (*diter)->z << ","
              << (*diter)->r << "} "
              //<< DISTANCE_SQUARED(*(*diter), *(dimensions[0])) << " "
		      << ((**diter)->dist2soma - dimension->dist2soma << " "
              << *(*viter) << std::endl;
  }
#endif
}

//TUAN: TODO challenge
//   how to check for 2 sites overlapping
//   if we don't retain the dimension's (x,y,z) coordinate
//  Even if we retain (x,y,z) this value change with the #capsule per compartment
//   and geometric sampling --> so not a good choice
bool HodgkinHuxleyVoltageJunction::checkSite(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset,
    CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset)
{
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  TissueSite& site = CG_inAttrPset->site;
  bool rval = (site.r == 0);
  if (!rval) rval = ((site.r * site.r) >= DISTANCE_SQUARED(site, *dimension));
  return rval;
}

bool HodgkinHuxleyVoltageJunction::confirmUniqueDeltaT(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset,
    CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}

HodgkinHuxleyVoltageJunction::~HodgkinHuxleyVoltageJunction() {}
