// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
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
#include <cmath>

#include "SegmentDescriptor.h"
#include "GlobalNTSConfig.h"

#define SMALL 1.0E-6
#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

//DEBUG ONLY when Vclamp type=3 is used
//#define DEBUG_UNCOUPLE_CHANNEL_AND_USE_DIRECT_VOLTAGE_CLAMP

SegmentDescriptor HodgkinHuxleyVoltageJunction::_segmentDescriptor;

// Get biomembrane surface area at the compartment i-th 
dyn_var_t HodgkinHuxleyVoltageJunction::getArea() // Tuan: check ok
{
  dyn_var_t area= 0.0;
#if defined (USE_SOMA_AS_POINT)
  area = 1.0; // [um^2]
#else
  area = dimensions[0]->surface_area;
#endif
  return area;
}


// GOAL: get all axial resistance from the explicit junction to the 
//    distal-cpt on parent branch
//    proximal-cpt(s) on children branch(es)
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
  unsigned numCpts = branchData->size;  //# of compartments
  SegmentDescriptor segmentDescriptor;
  assert(Vnew.size() == 1);
  assert(dimensions.size() == 1);


#ifdef IDEA_DYNAMIC_INITIALVOLTAGE
  dyn_var_t Vm_default = Vnew[0];
  for (unsigned int i=0; i<numCpts; ++i) {
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
    printf(" --> Area (of the 'soma' of the second .swc file)= %lf\n", area);
    // std::cerr << "area: " << area << std::endl;
  }
#endif

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
      Rb = ((*diter)->r ) ;
#ifdef USE_SCALING_NECK_FROM_SOMA
      //Rb = ((*diter)->r ) * 1.5;  //scaling factor 1.5 means the bigger interface with soma
      //  NOTE: should be applied for Axon hillock only
      //TEST 
      Rb /= SCALING_NECK_FROM_SOMA_WITH;
      //END TEST
#endif

#ifdef USE_SOMA_AS_ISOPOTENTIAL
      distance = std::fabs((*diter)->dist2soma - dimension->r); // SOMA is treated as a point source
#else
      //distance = (*diter)->dist2soma + dimension->r;
      distance = std::fabs((*diter)->dist2soma - dimension->dist2soma); //NOTE: The dist2soma of the first compartment stemming
      // from soma is always the distance from the center of soma to the center
      // of that compartment
#ifdef USE_STRETCH_SOMA_RADIUS
      //TEST 
      distance += STRETCH_SOMA_WITH;
      //  distance += 50.0;//TUAN TESTING - make soma longer
      //distance = std::fabs(b->r + a->dist2soma);
      //END TEST
#endif
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
    //NOTE:Based on TissueFunctor's connection order; the last voltageInputs[last]  comes from proximal-branch
  }
  if (getSharedMembers().deltaT)
  {// dt/2 jump
    cmt = 2.0 * Cm / *(getSharedMembers().deltaT);
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

//GOAL: predict Vnew[0] at offset time (t+dt/2) - Crank-Nicolson predictor-corrector scheme
//     using Vbranch(t) and Vnew[0](t)
void HodgkinHuxleyVoltageJunction::predictJunction(RNG& rng)
{
#ifdef DEBUG_UNCOUPLE_CHANNEL_AND_USE_DIRECT_VOLTAGE_CLAMP
  //Here, Vnew[0] and Vcur  does NOT change based on channel's current
  //instead, Vnew[0] is updated via VoltageClamp type=2 or type=3
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
    return;
#endif
  //TUAN DEBUG
#ifdef DEBUG_COMPARTMENT
  volatile int nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile int bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile int iteration = getSimulation().getIteration();
#endif
  //END TUAN DEBUG
  
  //element-1
  dyn_var_t conductance = cmt;
  //dyn_var_t current = cmt * Vcur; //here Vcur and Vnew[0] are supposed to be the same
  dyn_var_t current = cmt * Vnew[0]; //the reason to use this is to enable the trick in VoltageClamp type=2

  //element-2
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
    conductance += gloc; // at time (t+dt/2)
    current += gloc * (*(citer->reversalPotentials))[0];
  }

  //  1.b. ionic currents using GHK equations (-Iion)
  Array<ChannelCurrentsGHK>::iterator iiter = channelCurrentsGHK.begin();
  Array<ChannelCurrentsGHK>::iterator iend = channelCurrentsGHK.end();
  for (; iiter != iend; iiter++)
  {//IMPORTANT: subtraction is used
    current -=  (*(iiter->currents))[0]; //[pA/um^2]
#ifdef CONSIDER_DI_DV
      //take into account di/dv * Delta_V
      //IMPORTANT: addition is used
      ////TODO IMPORTANT
      //RHS[i] += di_dv * Vcur[i]; 
      //Aii[i] += di_dv;  
      //current += (*(iiter->di_dv))[i] * Vcur; //[pA/um^2]
      current += (*(iiter->di_dv))[0] * Vnew[0]; //[pA/um^2] - again Vnew[0] and Vcur still the same
      conductance +=  (*(iiter->di_dv))[0]; //[pA/um^2]
#endif
  }

  //  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<dyn_var_t*>::iterator iter = receptorReversalPotentials.begin();
  Array<dyn_var_t*>::iterator end = receptorReversalPotentials.end();
  Array<dyn_var_t*>::iterator giter = receptorConductances.begin();
  for (; iter != end; ++iter, ++giter)
  {
    conductance += **giter; //at time (t+dt/2)
    current += **iter * **giter;
  }

  //  3. synapse receptor currents using GHK type equations
  //  NOTE: Not available
  //{
  //  Array<ReceptorCurrentsGHK>::iterator riter = receptorCurrentsGHK.begin();
  //  Array<ReceptorCurrentsGHK>::iterator rend = receptorCurrentsGHK.end();
  //  for (; riter != rend; riter++)
  //  {//IMPORTANT: subtraction is used
  //     int i = riter->index; 
  //    RHS[i] -=  (*(riter->currents)); //[pA/um^2]
  //#ifdef CONSIDER_DI_DV
  //        //take into account di/dv * Delta_V
  //        //IMPORTANT: addition is used
  //        ////TODO IMPORTANT
  //        //RHS[i] += di_dv * Vcur[i]; 
  //        //Aii[i] += di_dv;  
  //        RHS[i] +=  (*(riter->di_dv))[i] * Vcur[i]; //[pA/um^2]
  //        Aii[i] +=  (*(riter->di_dv))[i]; //[pA/um^2]
  //#endif
  //  }
  //}

  //  4. injected currents [pA]
  iter = injectedCurrents.begin();
  end = injectedCurrents.end();
#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
  bool found_change = false;
  //  4b. consider abrupt change in injected currents [pA]
  Array<dyn_var_t>::iterator piter;
  piter = previous_injectedCurrent.begin();
#endif
  for (; iter != end; ++iter
#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
      , ++piter
#endif
      )
  {
#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        && gAxial.size() > 2 /* to skip soma head*/)
    {
#if 1 // use this only
#define STIM_CHANGE_THRESHOLD 10.0 // pA
      if (std::fabs(**iter - *piter) > STIM_CHANGE_THRESHOLD )
      {
        found_change = true;
        _count_timer = 1; //3; //6; //10;
      }
    #ifdef INJECTED_CURRENT_IS_POINT_PROCESS 
        current += **iter;
    #else
        current += **iter / area; // at time (t+dt/2)
    #endif
#else
        std::cout << " GET INTO WRONG LOCATION " << std::endl;
     #ifdef INJECTED_CURRENT_IS_POINT_PROCESS 
         current += **iter + ((**iter - *piter)/0.001 * Vcur);
     #else
         current += (**iter + ((**iter - *piter)/0.001 * Vcur)) / area; // at time (t+dt/2)
     #endif
#endif
         *piter = **iter;
    }
#else
    {
    #ifdef INJECTED_CURRENT_IS_POINT_PROCESS 
        current += **iter;
    #else
        current += **iter / area; // at time (t+dt/2)
    #endif
    }
#endif
  }

#if 0 
//TUAN: test --> the current injection should be applied to changing Vm
//  before it is taken out by diffusion
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
  {
#if not defined(PREDICT_JUNCTION_IGNORE_AXIAL)
    // 5. Current loss due to passive diffusion to adjacent compartments
    Array<dyn_var_t>::iterator xiter = gAxial.begin(), xend = gAxial.end();
    Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
    for (; xiter != xend; ++xiter, ++viter)
    {
#if 0
      //original approach
      //current += (*xiter) * ((**viter) - Vcur);
      current += (*xiter) * ((**viter) - Vnew[0]);
#else
      current += (*xiter) * (**viter);
      conductance += (*xiter);  
#endif
    }
#endif
  }
  else{
    // 5. Current loss due to passive diffusion to adjacent compartments
    Array<dyn_var_t>::iterator xiter = gAxial.begin(), xend = gAxial.end();
    Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
    for (; xiter != xend; ++xiter, ++viter)
    {
#if 0
      //original approach
      //current += (*xiter) * ((**viter) - Vcur);
      current += (*xiter) * ((**viter) - Vnew[0]);
#else
      current += (*xiter) * (**viter);
      conductance += (*xiter);  
#endif
    }
  }
#else

#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
  if (not found_change and not (_count_timer > 0))
  {
    if (_count_timer >= 0)
      _count_timer -= 1;
#endif
    // 5. Current loss due to passive diffusion to adjacent compartments
    Array<dyn_var_t>::iterator xiter = gAxial.begin(), xend = gAxial.end();
    Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
    for (; xiter != xend; ++xiter, ++viter)
    {
#if 0
      //original approach
      //current += (*xiter) * ((**viter) - Vcur);
      current += (*xiter) * ((**viter) - Vnew[0]);
#else
      current += (*xiter) * (**viter);
      conductance += (*xiter);  
#endif
    }
#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
  }
#endif
#endif


  //float Vold = Vnew[0];
  Vnew[0] = current / conductance; //estimate at (t+dt/2)

#ifdef DEBUG_ASSERT
  if (not (Vnew[0] == Vnew[0])
      //			or std::fabs(Vnew[0]-Vold)/(*getSharedMembers().deltaT)
      or Vnew[0]> 230.0
      or Vnew[0] < -330.0
     )
  {
    printDebugHH();
    assert(0);
    assert(Vnew[0] == Vnew[0]);
  }
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

//GOAL: recalculate using an updated 'Vnew', i.e. correct Vnew[0] at (t+dt/2) 
// and finally update Vcur, and Vnew[0] at (t+dt)
// NOTE: at entry, Vcur is not the same as Vnew[0]
void HodgkinHuxleyVoltageJunction::correctJunction(RNG& rng)
{
#ifdef DEBUG_UNCOUPLE_CHANNEL_AND_USE_DIRECT_VOLTAGE_CLAMP
  //Here, Vnew[0] and Vcur  does NOT change based on channel's current
  //instead, Vnew[0] is updated via VoltageClamp type=2 or type=3
  //Vcur = Vnew[0] = 2.0 * Vnew[0] - Vcur;
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
    return;
#endif
  //TUAN DEBUG
#ifdef DEBUG_COMPARTMENT
  volatile int nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile int bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile int iteration = getSimulation().getIteration();
#endif
  //END TUAN DEBUG
 
  //element-1
  dyn_var_t conductance = cmt;
  dyn_var_t current = cmt * Vcur; //dont' use Vnew[0] 

  //element-2
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
    conductance += gloc; //should be at time (t+dt/2)
    current += gloc * (*(citer->reversalPotentials))[0];
  }

  //  1.b. ionic currents using GHK equations (-Iion)
  Array<ChannelCurrentsGHK>::iterator iiter = channelCurrentsGHK.begin();
  Array<ChannelCurrentsGHK>::iterator iend = channelCurrentsGHK.end();
  for (; iiter != iend; iiter++)
  {//IMPORTANT: subtraction is used
    current -=  (*(iiter->currents))[0]; //[pA/um^2]
#ifdef CONSIDER_DI_DV
      //take into account di/dv * Delta_V
      //IMPORTANT: addition is used
      current += (*(iiter->di_dv))[0] * Vcur; //[pA/um^2] - don't use Vnew[0]
      conductance +=  (*(iiter->di_dv))[0]; //[pA/um^2]
#endif
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
  //{
  //  Array<ReceptorCurrentsGHK>::iterator riter = receptorCurrentsGHK.begin();
  //  Array<ReceptorCurrentsGHK>::iterator rend = receptorCurrentsGHK.end();
  //  for (; riter != rend; riter++)
  //  {//IMPORTANT: subtraction is used
  //     int i = riter->index; 
  //    RHS[i] -=  (*(riter->currents)); //[pA/um^2]
  //#ifdef CONSIDER_DI_DV
  //        //take into account di/dv * Delta_V
  //        //IMPORTANT: addition is used
  //        ////TODO IMPORTANT
  //        current += (*(riter->di_dv))[0] * Vcur; //[pA/um^2] - don't use Vnew[0]
  //        conductance +=  (*(riter->di_dv))[0]; //[pA/um^2]
  //#endif
  //  }
  //}

  //  4. injected currents [pA]
  iter = injectedCurrents.begin();
  end = injectedCurrents.end();
#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
  bool found_change = false;
  //  4b. consider abrupt change in injected currents [pA]
  Array<dyn_var_t>::iterator piter;
  piter = previous_injectedCurrent.begin();
#endif
  for (; iter != end; ++iter
#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
      , ++piter
#endif
      )
  {
#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
    if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
        && gAxial.size() > 2 /* to skip soma head*/)
    {
#if 1 // use this only
#define STIM_CHANGE_THRESHOLD 10.0 // pA
      //if (std::fabs(**iter - *piter) > STIM_CHANGE_THRESHOLD )
      //{
      //  found_change = true;
      //  _count_timer = 10;
      //}
    #ifdef INJECTED_CURRENT_IS_POINT_PROCESS 
        current += **iter;
    #else
        current += **iter / area; // at time (t+dt/2)
    #endif
#else
        std::cout << " GET INTO WRONG LOCATION " << std::endl;
     #ifdef INJECTED_CURRENT_IS_POINT_PROCESS 
         current += **iter + ((**iter - *piter)/0.001 * Vcur);
     #else
         current += (**iter + ((**iter - *piter)/0.001 * Vcur)) / area; // at time (t+dt/2)
     #endif
#endif
         *piter = **iter;
    }
#else
    {
    #ifdef INJECTED_CURRENT_IS_POINT_PROCESS 
        current += **iter;
    #else
        current += **iter / area; // at time (t+dt/2)
    #endif
    }
#endif
  }

  // 5. Current loss due to passive diffusion to adjacent compartments
  Array<dyn_var_t>::iterator xiter = gAxial.begin(), xend = gAxial.end();
  Array<dyn_var_t*>::iterator viter = voltageInputs.begin();
  //int i =0;
  for (; xiter != xend; ++xiter, ++viter)
  {
    //i++;
    current += (*xiter) * (**viter); //using V(t+dt/2) from adjacent compartments
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
void HodgkinHuxleyVoltageJunction::add_zero_didv(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset)
{
  _zero_conductance = 0;
  injectedCurrents_conductance_didv.push_back(&_zero_conductance);
}

#ifdef CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
void HodgkinHuxleyVoltageJunction::update_stim_reference(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageJunctionInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageJunctionOutAttrPSet* CG_outAttrPset)
{
  //this kep track the value of Istim in the previous time-step 
  dyn_var_t value = 0.0; 
  previous_injectedCurrent.push_back(value);
}
#endif

//#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
#if defined(CONSIDER_MANYSPINE_EFFECT_OPTION1) || defined(CONSIDER_MANYSPINE_EFFECT_OPTION2_revised)
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
