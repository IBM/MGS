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
#include "Branch.h"

//#define DEBUG_HH
#include <iomanip>
#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

SegmentDescriptor HodgkinHuxleyVoltageJunction::_segmentDescriptor;

// Get biomembrane surface area at the compartment i-th 
dyn_var_t HodgkinHuxleyVoltageJunction::getArea() // Tuan: check ok
{
  dyn_var_t area= 0.0;
  area = dimensions[0]->surface_area;
	return area;
}


void HodgkinHuxleyVoltageJunction::initializeJunction(RNG& rng)
{ // explicit junction (which can be soma (with branches are axon/dendrite
  // trees)
  // or a cut point junction 
	// or a branching point junction with 
  //    either 3 or more branches (one from main, 2+ for children
  // branches)
  //    or 2 branches (one from main, one from children when there is branchType changing, 
  //    but not branchpoint), e.g. from prox-axon to AIS, or AIS to distal-axon on 1 branch
  //               or apical-trunk to apical-tuft
  // )
  unsigned size = branchData->size;  //# of compartments
  SegmentDescriptor segmentDescriptor;
  assert(Vnew.size() == 1);
  assert(dimensions.size() == 1);


#ifdef IDEA_DYNAMIC_INITIALVOLTAGE
  dyn_var_t Vm_default = Vnew[0];
  for (unsigned int i=0; i<size; ++i) {
    if (Vm_dists.size() > 0) {
      unsigned int j;
      //NOTE: 'n' bins are splitted by (n-1) points
      if (Vm_values.size() - 1 != Vm_dists.size())
      {
        std::cerr << "Vm_values.size = " << Vm_values.size() 
          << "; Vm_dists.size = " << Vm_dists.size() << std::endl; 
      }
      assert(Vm_values.size() -1 == Vm_dists.size());
      for (j=0; j<Vm_dists.size(); ++j) {
        if ((dimensions)[i]->dist2soma < Vm_dists[j]) break;
      }
      Vnew[i] = Vm_values[j];
    }
		else if (Vm_branchorders.size() > 0)
		{
      unsigned int j;
      assert(Vm_values.size() == Vm_branchorders.size());
      SegmentDescriptor segmentDescriptor;
      for (j=0; j<Vm_branchorders.size(); ++j) {
        if (segmentDescriptor.getBranchOrder(branchData->key) == Vm_branchorders[j]) break;
      }
			if (j == Vm_branchorders.size() and Vm_branchorders[j-1] == GlobalNTS::anybranch_at_end)
			{
				Vnew[i] = Vm_values[j-1];
			}
			else if (j < Vm_values.size()) 
        Vnew[i] = Vm_values[j];
      else
        Vnew[i] = Vm_default;
		}
		else {
      Vnew[i] = Vm_default;
    }
  }
#endif
  Vcur = Vnew[0];
  // So, one explicit junction is composed of one compartment 
  // which can be explicit cut-point junction or
  //              explicit branching-point junction
  DimensionStruct* dimension = dimensions[0];  
                                             
  area = getArea();
#ifdef DEBUG_HH
  // check 'bouton' neuron ???
  if (_segmentDescriptor.getNeuronIndex(branchData->key) == 2)
  {
    printf(" --> Area = %lf\n", area);
    // std::cerr << "area: " << area << std::endl;
  }
#endif

	//NOTE: Should not use the whole area here
  dyn_var_t Poar = M_PI / (area * getSharedMembers().Ra);  // Pi-over-(area *
                                                           // axial-resistance)
  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin(),
                                    dend = dimensionInputs.end();
  for (; diter != dend; ++diter)
  {
	  //NOTE: if the junction is the SOMA, we should not use the radius of the SOMA
	  //      in calculating the cross-sectional area
	  dyn_var_t Rb;
	  dyn_var_t distance ;
	  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
	  {
		  //Rb = ((*diter)->r ) * 1.5;  //scaling factor 1.5 means the bigger interface with soma
      //  NOTE: should be applied for Axon hillock only
		  Rb = ((*diter)->r ) ;
      //TEST 
			Rb /= SCALING_NECK_FROM_SOMA;
      //END TEST
#ifdef USE_SOMA_AS_POINT
      distance = (*diter)->dist2soma - dimension->r; // SOMA is treated as a point source
#else
      //distance = (*diter)->dist2soma + dimension->r;
      distance = (*diter)->dist2soma; //NOTE: The dist2soma of the first compartment stemming
         // from soma is always the distance from the center of soma to the center
         // of that compartment
      //distance += 50.0;//TUAN TESTING - make soma longer
      //TEST 
      distance += STRETCH_SOMA_WITH;
      //END TEST
#endif
      assert(distance > 0);
	  }else{
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
		  Rb = ((*diter)->r); //the small diameter of the branch means small current pass to it
#else
		  Rb = 0.5 * ((*diter)->r + dimension->r);
#endif
	    distance = fabs((*diter)->dist2soma - dimension->dist2soma);
      assert(distance > 0);
	  }
//#define TEST_IDEA_R_HALF_SOMA
//#ifdef TEST_IDEA_R_HALF_SOMA
//		  Rb = 0.5 * ((*diter)->r + dimension->r);
//#endif
    if (distance <= 0) 
      std::cerr << "distance = " << distance << std::endl;
	  assert(distance > 0);
	  gAxial.push_back(Poar * Rb * Rb / distance);
  }
  if (getSharedMembers().deltaT)
  {
    cmt = 2.0 * Cm / *(getSharedMembers().deltaT);
    //cmt =  Cm / *(getSharedMembers().deltaT);
  }
#ifdef DEBUG_HH
  std::cerr << "JUNCTION (" << dimension->x << "," << dimension->y << ","
		<< dimension->z << "," << dimension->r 
		<< "," << dimension->surface_area 
		<< "," << dimension->dist2soma
		<< "," << dimension->length
		<< ")" << std::endl;
#endif
#ifdef DEBUG_ASSERT
  if (cmt <= 0)
  {
	  std::cout << "HINTS: Check CptParams...par file, maybe you're using the neurons with MTYPE"
		  << " not defined in the param file\n";
	  assert(cmt > 0);
  }
#endif
}

//GOAL: predict Vnew[0] at offset time (n+1/2) - Crank-Nicolson predictor-corrector scheme
void HodgkinHuxleyVoltageJunction::predictJunction(RNG& rng)
{
  //TUAN DEBUG
#ifdef DEBUG_COMPARTMENT
  volatile int nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile int bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile int iteration = getSimulation().getIteration();
#endif
  //END TUAN DEBUG
  dyn_var_t conductance = cmt;
  //dyn_var_t current = cmt * Vcur;
  dyn_var_t current = cmt * Vnew[0];

  conductance += gLeak;
  current += gLeak * getSharedMembers().E_leak;

	/* * * Sum Currents * * */
	// loop through different kinds of currents (Kv, Nav1.6, ...)
	//  1.a. ionic currents using Hodgkin-Huxley type equations (+g*Erev)
  Array<ChannelCurrents>::iterator citer = channelCurrents.begin();
  Array<ChannelCurrents>::iterator cend = channelCurrents.end();
  for (; citer != cend; ++citer)
  {
    dyn_var_t gloc = (*(citer->conductances))[0];
    conductance += gloc;
    current += gloc * (*(citer->reversalPotentials))[0];
  }

	//  1.b. ionic currents using GHK equations (-Iion)
	Array<ChannelCurrentsGHK>::iterator iiter = channelCurrentsGHK.begin();
	Array<ChannelCurrentsGHK>::iterator iend = channelCurrentsGHK.end();
	for (; iiter != iend; iiter++)
	{//IMPORTANT: subtraction is used
		current -=  (*(iiter->currents))[0]; //[pA/um^2]
	}

	//  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<dyn_var_t*>::iterator iter = receptorReversalPotentials.begin();
  Array<dyn_var_t*>::iterator end = receptorReversalPotentials.end();
  Array<dyn_var_t*>::iterator giter = receptorConductances.begin();
  for (; iter != end; ++iter, ++giter)
  {
    conductance += **giter;
    current += **iter * **giter;
  }

	//  3. synapse receptor currents using GHK type equations (gV, gErev)
	//  NOTE: Not available

  //  4. injected currents
  iter = injectedCurrents.begin();
  end = injectedCurrents.end();
  for (; iter != end; ++iter)
  {
    current += **iter / area;
  }

	// 5. Current loss due to passive diffusion to adjacent compartments
  Array<dyn_var_t>::iterator xiter = gAxial.begin(), xend = gAxial.end();
  Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    current += (*xiter) * ((**viter) - Vcur);
  }

	//float Vold = Vnew[0];
  Vnew[0] = current / conductance; //estimate at (t+dt/2)

#ifdef DEBUG_ASSERT
  if (not (Vnew[0] == Vnew[0])
//			or std::fabs(Vnew[0]-Vold)/(*getSharedMembers().deltaT)
//			or Vnew[0]> 130.0
		 )
			{
	  printDebugHH();
  }
	assert(Vnew[0] == Vnew[0]);
#endif

#ifdef DEBUG_HH
  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
            << " JUNCTION PREDICT"
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
            << Vnew[0] << std::endl;
#endif
}

//GOAL: correct Vnew[0] at (t+dt/2) 
// and finally update at (t+dt) for Vcur, and Vnew[0]
void HodgkinHuxleyVoltageJunction::correctJunction(RNG& rng)
{
  dyn_var_t conductance = cmt;
  dyn_var_t current = cmt * Vcur;
  //TUAN DEBUG
#ifdef DEBUG_COMPARTMENT
  volatile int nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile int bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile int iteration = getSimulation().getIteration();
#endif
  //END TUAN DEBUG
 
  conductance += gLeak;
  current += gLeak * getSharedMembers().E_leak;

	//  1.a. ionic currents using Hodgkin-Huxley type equations (+g*Erev)
  Array<ChannelCurrents>::iterator citer = channelCurrents.begin();
  Array<ChannelCurrents>::iterator cend = channelCurrents.end();
  for (; citer != cend; ++citer)
  {
    dyn_var_t gloc = (*(citer->conductances))[0];
    conductance += gloc;
    current += gloc * (*(citer->reversalPotentials))[0];
  }

	//  1.b. ionic currents using GHK equations (-Iion)
	Array<ChannelCurrentsGHK>::iterator iiter = channelCurrentsGHK.begin();
	Array<ChannelCurrentsGHK>::iterator iend = channelCurrentsGHK.end();
	for (; iiter != iend; iiter++)
	{//IMPORTANT: subtraction is used
		current -=  (*(iiter->currents))[0]; //[pA/um^2]
	}

	//  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<dyn_var_t*>::iterator iter = receptorReversalPotentials.begin();
  Array<dyn_var_t*>::iterator end = receptorReversalPotentials.end();
  Array<dyn_var_t*>::iterator giter = receptorConductances.begin();
  for (; iter != end; ++iter, ++giter)
  {
    conductance += **giter;
    current += **iter * **giter;
  }

	//  3. synapse receptor currents using GHK type equations (gV, gErev)
	//  NOTE: Not available

  //  4. injected currents [pA]
  iter = injectedCurrents.begin();
  end = injectedCurrents.end();
  for (; iter != end; ++iter)
  {
    current += **iter / area;
  }

  Array<dyn_var_t>::iterator xiter = gAxial.begin(), xend = gAxial.end();
  Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
  int i =0;
  for (; xiter != xend; ++xiter, ++viter)
  {
    i++;
    current += (*xiter) * (**viter);
    conductance += (*xiter);
  }

  Vnew[0] = current / conductance;

  // This is the swap phase
  Vcur = Vnew[0] = 2.0 * Vnew[0] - Vcur;

#ifdef DEBUG_ASSERT
  if (not (Vnew[0] == Vnew[0])){
    std::cerr << "Iteration: " << getSimulation().getIteration() << std::endl;
	  printDebugHH();
  }
	assert(Vnew[0] == Vnew[0]);
#endif

#ifdef DEBUG_HH
	printDebugHH();
#endif
}

void HodgkinHuxleyVoltageJunction::printDebugHH(std::string phase)
{
	std::cerr << "step,time|" << phase << " [rank,nodeIdx,instanceIdx] " <<
		"(neuronIdx,branchIdx,brchOrder,brType){x,y,z,r | dist2soma,surfarea,volume,len} Vm" << std::endl;
  std::cerr << getSimulation().getIteration() << "," 
    <<  getSimulation().getIteration() * *getSharedMembers().deltaT
    << "|" << phase 
    << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
    << getIndex() << "] "
    << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchOrder(branchData->key) << ","
    << _segmentDescriptor.getBranchType(branchData->key) << ") {"
    << dimensions[0]->x << "," 
    << dimensions[0]->y << ","
    << dimensions[0]->z << "," 
    << dimensions[0]->r << " | " 
    << dimensions[0]->dist2soma << "," << dimensions[0]->surface_area << ","
    << dimensions[0]->volume << "," << dimensions[0]->length << "} " 
    << Vnew[0]
    << std::endl;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin();
  Array<dyn_var_t*>::iterator vend = voltageInputs.end();
  int c = -1;

	std::cerr << "JCT_INPUT_i " <<
		"(neuronIdx,branchIdx,brchOrder, brType, COMPUTEORDER){x,y,z,r | dist2soma,surfarea,volume,len} Vm" << std::endl;
  Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
  for (viter = voltageInputs.begin(); viter != vend; ++viter, ++diter)
  {
    c++;
    std::cerr << " JCT_INPUT_" << c 
      << "(" << _segmentDescriptor.getNeuronIndex(branchDataInputs[c]->key) << ","
      << std::setw(2) << _segmentDescriptor.getBranchIndex(branchDataInputs[c]->key) << ","
      << _segmentDescriptor.getBranchOrder(branchDataInputs[c]->key) << ","
      << _segmentDescriptor.getBranchType(branchDataInputs[c]->key) << ","
      << _segmentDescriptor.getComputeOrder(branchDataInputs[c]->key) << ") {"
      << std::setprecision(3) << (*diter)->x << "," 
      << std::setprecision(3) << (*diter)->y << "," 
      << std::setprecision(3) << (*diter)->z << ","
      << std::setprecision(3) << (*diter)->r << " | " 
      << (*diter)->dist2soma << "," << (*diter)->surface_area << "," 
      << (*diter)->volume << "," << (*diter)->length  << "} "
      //<< DISTANCE_SQUARED(*(*diter), *(dimensions[0])) << " "
      //<< (*diter)->dist2soma - (dimensions[0])->dist2soma << " "
      << *(*viter) << std::endl;
  }
	
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

#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
void HodgkinHuxleyVoltageJunction::updateSpineCount(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset) 
{
  unsigned size = branchData->size;  //# of compartments
  if (countSpineConnected.size() != size) 
  {
    countSpineConnected.increaseSizeTo(size);
    for (int i = 0; i < size; i++)
      countSpineConnected[i] = 0;
  }
  countSpineConnected[0]++;
}

void HodgkinHuxleyVoltageJunction::updateGapJunctionCount(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset) 
{
  unsigned size = branchData->size;  //# of compartments
  if (countGapJunctionConnected.size() != size) 
  {
    countGapJunctionConnected.increaseSizeTo(size); 
    for (int i = 0; i < size; i++)
      countGapJunctionConnected[i] = 0;
  }
  countGapJunctionConnected[0]++;
}
#endif
