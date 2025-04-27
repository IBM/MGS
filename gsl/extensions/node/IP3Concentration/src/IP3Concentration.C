// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "IP3Concentration.h"
#include "CG_IP3Concentration.h"
#include "rndm.h"
#include "GridLayerDescriptor.h"
#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"
#include "StringUtils.h"
#include "Params.h"

#include <iomanip>
#include <cmath>

#define SMALL 1.0E-6
#define DISTANCE_SQUARED(a, b)               \
  ((((a)->x - (b)->x) * ((a)->x - (b)->x)) + \
   (((a)->y - (b)->y) * ((a)->y - (b)->y)) + \
   (((a)->z - (b)->z) * ((a)->z - (b)->z)))

SegmentDescriptor IP3Concentration::_segmentDescriptor;

// NOTE: value = 1e6/(zIP3*Farad)
// zIP3 = valence of IP3
// Farad = Faraday's constant
#define uM_um_cubed_per_pA_msec 5.18213484752067

#define isProximalCase0 (proximalDimension == 0)  // no flux boundary condition
#define isProximalCase1 \
  (proximalJunction == 0 && proximalDimension != 0)  // connected to proximal
                                                     // cut or branch point for
                                                     // implicit solve
#define isProximalCase2 (proximalJunction)  // connected to proximal junction

#define isDistalCase0 \
  (distalDimensions.size() == 0)  // no flux boundary condition
#define isDistalCase1 \
  (distalAiis.size() == 1)  // connected to distal cut point for implicit solve
#define isDistalCase2        \
  (distalAiis.size() == 0 && \
   distalInputs.size() == 1)  // connected to distal explicit junction
#define isDistalCase3  \
  (distalAiis.size() > \
   1)  // connected to distal branch point for implicit solve

#if IP3_CYTO_DYNAMICS == FAST_BUFFERING
#define D_IP3 (getSharedMembers().D_IP3eff)
#else
#define D_IP3 (getSharedMembers().D_IP3)
#endif

//#define DEBUG_HH
// Conserved region (only change ClassName)
//{{{
void IP3Concentration::solve(RNG& rng)
{
  if (computeOrder == 0)
  {
    doForwardSolve();
    doBackwardSolve();
  }
#ifdef DEBUG_HH
	std::cerr << "Solve:\n";
	printDebugHH();
#endif
}

#if MAX_COMPUTE_ORDER > 0
void IP3Concentration::forwardSolve1(RNG& rng)
{
  if (computeOrder == 1)
  {
    doForwardSolve();
  }
}

void IP3Concentration::backwardSolve1(RNG& rng)
{
  if (computeOrder == 1) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 1
void IP3Concentration::forwardSolve2(RNG& rng)
{
  if (computeOrder == 2)
  {
    doForwardSolve();
  }
}

void IP3Concentration::backwardSolve2(RNG& rng)
{
  if (computeOrder == 2) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 2
void IP3Concentration::forwardSolve3(RNG& rng)
{
  if (computeOrder == 3)
  {
    doForwardSolve();
  }
}

void IP3Concentration::backwardSolve3(RNG& rng)
{
  if (computeOrder == 3) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 3
void IP3Concentration::forwardSolve4(RNG& rng)
{
  if (computeOrder == 4)
  {
    doForwardSolve();
  }
}

void IP3Concentration::backwardSolve4(RNG& rng)
{
  if (computeOrder == 4) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 4
void IP3Concentration::forwardSolve5(RNG& rng)
{
  if (computeOrder == 5)
  {
    doForwardSolve();
  }
}

void IP3Concentration::backwardSolve5(RNG& rng)
{
  if (computeOrder == 5) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 5
void IP3Concentration::forwardSolve6(RNG& rng)
{
  if (computeOrder == 6)
  {
    doForwardSolve();
  }
}

void IP3Concentration::backwardSolve6(RNG& rng)
{
  if (computeOrder == 6) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 6
void IP3Concentration::forwardSolve7(RNG& rng)
{
  if (computeOrder == 7)
  {
    doForwardSolve();
  }
}

void IP3Concentration::backwardSolve7(RNG& rng)
{
  if (computeOrder == 7) doBackwardSolve();
}
#endif

bool IP3Concentration::confirmUniqueDeltaT(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
    CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}

//TUAN: TODO challenge
//   how to check for 2 sites overlapping
//   if we don't retain the dimension's (x,y,z) coordinate
//  Even if we retain (x,y,z) this value change with the #capsule per compartment
//   and geometric sampling --> so not a good choice
bool IP3Concentration::checkSite(const CustomString& CG_direction,
                                const CustomString& CG_component,
                                NodeDescriptor* CG_node, Edge* CG_edge,
                                VariableDescriptor* CG_variable,
                                Constant* CG_constant,
                                CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
                                CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset)
{
  TissueSite& site = CG_inAttrPset->site;
  bool atSite = (site.r == 0);
  for (unsigned int i = 0; !atSite && i < dimensions.size(); ++i)
    atSite = ((site.r * site.r) >= DISTANCE_SQUARED(&site, dimensions[i]));
  return atSite;
}

void IP3Concentration::setProximalJunction(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
    CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset)
{
  proximalJunction = true;
}

// update: IP3(t+dt) = 2 * IP3(t+dt/2) - IP3(t)
// second-step (final step) in Crank-Nicolson method
void IP3Concentration::finish(RNG& rng)
{
  unsigned size = branchData->size;
#ifdef DEBUG_HH
  printDebugHH();
#endif
  for (int i = 0; i < size; ++i)
  {
    IP3_cur[i] = IP3_new[i] = 2.0 * IP3_new[i] - IP3_cur[i];
#ifdef DEBUG_ASSERT
    if (IP3_new[i] != IP3_new[i] or 
        IP3_new[i] <= 0)
      printDebugHH();
    assert(IP3_new[i] >= 0);
    assert(IP3_new[i] == IP3_new[i]);  // making sure IP3_new[i] is not NaN
#endif
  }
}

// Get cytoplasmic surface area (um^2)
// at the compartment based on its index 'i'
dyn_var_t IP3Concentration::getArea(int i) // Tuan: check ok
{
  dyn_var_t area= 0.0;
  area = dimensions[i]->surface_area * FRACTION_SURFACEAREA_CYTO;
  return area;
}

// Get cytoplasmic volume (um^3) 
// at the compartment based on its index 'i'
dyn_var_t IP3Concentration::getVolume(int i) // Tuan: check ok
{
  dyn_var_t volume = 0.0;
  volume = dimensions[i]->volume * FRACTIONVOLUME_CYTO;
  return volume;
}
//}}} //end Conserved region

// GOAL: initialize data at each branch
//    the compartments along one branch are indexed from distal (index=0)
//    to the proximal (index=branchData->size-1)
//    so Aim[..] from distal side
//       Aip[..] from proximal side
void IP3Concentration::initializeCompartmentData(RNG& rng)
{
  // for a given computing process:
  //  here all the data in vector-form are initialized to
  //  the same size as the number of compartments in a branch (i.e. branchData)
  unsigned numCpts = branchData->size;  //# of compartments
  computeOrder = _segmentDescriptor.getComputeOrder(branchData->key);

  if (isProximalCase2) assert(computeOrder == 0);
  if (isDistalCase2) assert(computeOrder == MAX_COMPUTE_ORDER);
  assert(dimensions.size() == numCpts);
  assert(IP3_new.size() == numCpts);
  assert(distalDimensions.size() == distalInputs.size());

  // allocate data
  if (IP3_cur.size() != numCpts) IP3_cur.increaseSizeTo(numCpts);
  if (Aii.size() != numCpts) Aii.increaseSizeTo(numCpts);
  if (Aip.size() != numCpts) Aip.increaseSizeTo(numCpts);
  if (Aim.size() != numCpts) Aim.increaseSizeTo(numCpts);
  if (RHS.size() != numCpts) RHS.increaseSizeTo(numCpts);
  if (currentDensityToConc.size() != numCpts) currentDensityToConc.increaseSizeTo(numCpts);

  // initialize data
  IP3_cur[0] = IP3_new[0];
  for (int i = 1; i < numCpts; ++i)
  {
    IP3_new[i] = IP3_new[0];
    IP3_cur[i] = IP3_cur[0];
  }
  // go through each compartments in a branch
  for (int i = 0; i < numCpts; ++i)
  {
    Aii[i] = Aip[i] = Aim[i] = RHS[i] = 0.0;
    currentDensityToConc[i] = getArea(i) * uM_um_cubed_per_pA_msec / getVolume(i);
  }

  // go through different kinds of injected IP3 currents
  //   one of which is the bidirectional current from spine neck
  Array<InjectedIP3Current>::iterator iiter = injectedIP3Currents.begin();
  Array<InjectedIP3Current>::iterator iend = injectedIP3Currents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < numCpts)
      iiter->currentToConc = uM_um_cubed_per_pA_msec / getVolume(iiter->index);
  }



  if (!isProximalCase0)
  {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
    if (isProximalCase1)
    {
      Aip[numCpts - 1] =
        -getLambda(dimensions[numCpts - 1], numCpts-1);  // [nS/um^2]
    }
    else{
      Aip[numCpts - 1] =
        -getLambda(dimensions[numCpts - 1], proximalDimension, numCpts-1, true);  // [nS/um^2]
    }
    assert(0);
#else
    Aip[numCpts - 1] =
        -getLambda_parent(dimensions[numCpts - 1], proximalDimension, numCpts-1, true);  // [nS/um^2]
#endif
  }

  if (isDistalCase1 || isDistalCase2)
  {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
    if (isDistalCase1)
      Aim[0] = -getLambda(dimensions[0], 0);
    else
      Aim[0] = -getLambda(dimensions[0],distalDimensions[0], 0, true);
    assert(0);
#else
    Aim[0] = -getLambda_child(dimensions[0],distalDimensions[0], 0, true);
#endif
  }

  for (int i = 1; i < numCpts; i++)
  {
    Aim[i] = -getLambda_child(dimensions[i], dimensions[i - 1], i);
  }

  for (int i = 0; i < numCpts - 1; i++)
  {
    Aip[i] = -getLambda_parent(dimensions[i], dimensions[i + 1], i);
  }

  /* FIX */
  if (isDistalCase3)
  {
    // Compute total volume of the junction...
    dyn_var_t volume = getVolume(0);

    // Compute Aij[n] for the junction...one of which goes in Aip[0]...
    if (numCpts == 1)
    {//branch has only 1 compartment, so get compartment in another branch
			// which is referenced via proximalDimension
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
      //CHECK AGAIN
      if (isProximalCase1)
      {
        Aip[0] =
          -getLambda(dimensions[0], i);  // [nS/um^2]
      }
      else{
        Aip[0] = -getAij(dimensions[0], proximalDimension, volume, true);
      }
      assert(0);
#else
      Aip[0] = -getAij_parent(dimensions[0], proximalDimension, volume, true);
#endif
    }
    else
    {
      Aip[0] = -getAij_parent(dimensions[0], dimensions[1], volume);
    }
    /* reverted back to original approach
  //IMPORTANT CHANGE:
  // Unlike the original approach
  //   which doesn't have a compartment for the implicit branching
  // The branch now has
  //at least 2: one compartment as implicit branching point + one as regular
  //    compartment-zero as implicit branching compartment
  //    compartment-1th and above as normal
      Aip[0] = -getAij(dimensions[1], dimensions[0], volume);
  */
    for (int n = 0; n < distalDimensions.size(); n++)
    {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
      //CHECK AGAIN
      Aij.push_back(-getAij(dimensions[0], distalDimensions[n], volume, true));
      assert(0);
#else
      Aij.push_back(-getAij_child(dimensions[0], distalDimensions[n], volume, true));
#endif
    }
  }

#ifdef SECOND_ORDER_SPATIAL
  if (isProximalCase0)
  {
    Aim[numCpts-1] *= 2; 
  }
  if (isDistalCase0)
  {
    Aip[0] *= 2; 
  }
#endif

#ifdef DEBUG_HH
  printDebugHH();
#endif
}

void IP3Concentration::printDebugHH()
{
  unsigned size = branchData->size;
  for (int i = 0; i < size; ++i)
  {
    this->printDebugHH(i);
  }
}

void IP3Concentration::printDebugHH(int cptIndex)
{
  unsigned size = branchData->size;
  if (cptIndex == 0)
  {
    std::cerr << "iter,time| BRANCH [rank, nodeIdx, layerIdx, cptIdx]"
      << "(neuronIdx, brIdx, brOrder, brType) distal(C0 | C1 | C2 | C3) :"
      << " prox( C0 | C1 | C2) |"
      << "{x,y,z,r, dist2soma, surface_area, volume, length} IP3\n";
  }
  int i  = cptIndex;
  std::cerr << getSimulation().getIteration() << "," <<
    dyn_var_t(getSimulation().getIteration()) *
    *getSharedMembers().deltaT << "| BRANCH"
    << " [" << getSimulation().getRank() << "," << getNodeIndex()
    << "," << getIndex() << "," << i << "] "
    << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
    << std::setw(2) << _segmentDescriptor.getBranchIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchOrder(branchData->key) << ","
    << _segmentDescriptor.getBranchType(branchData->key) << ") |"
    << isDistalCase0 << "|" << isDistalCase1 << "|" << isDistalCase2
    << "|" << isDistalCase3 << "|" << isProximalCase0 << "|"
    << isProximalCase1 << "|" << isProximalCase2 << "|"
    << " {" 
    << std::setprecision(3) << dimensions[i]->x << "," 
    << std::setprecision(3) << dimensions[i]->y << ","
    << std::setprecision(3) << dimensions[i]->z << "," 
    << std::setprecision(3) << dimensions[i]->r << " | " 
    << dimensions[i]->dist2soma  << ","
    << dimensions[i]->surface_area << "," 
    << dimensions[i]->volume << "," << dimensions[i]->length 
    << "} "
    << IP3_new[i]  << " " << std::endl;
}


// Update: RHS[], Aii[] at time (t+dt/2)
// Unit: RHS =  [uM/msec]
//       Aii =  [1/msec]
// Thomas algorithm forward step 
void IP3Concentration::doForwardSolve()
{
  unsigned size = branchData->size;

  //Find A[ii]i and RHS[ii]  
  //  1. ionic currents 
  for (int i = 0; i < size; i++)
  {
    RHS[i] = getSharedMembers().bmt * IP3_cur[i];
    if (i == 0)
    {
      if (isDistalCase3)
      {
        Aii[0] = getSharedMembers().bmt - Aip[0];
        for (int n = 0; n < distalInputs.size(); n++)
        {
          //initial assign
          Aii[0] -= Aij[n];

          //this is part of removing lower part of matrix
          Aii[0] -= Aij[n] * *distalAips[n] / *distalAiis[n];
          RHS[0] -= Aij[n] * *distalInputs[n] / *distalAiis[n];
        }
      }
      else{
        Aii[i] = getSharedMembers().bmt - Aim[i] - Aip[i];
        if (isDistalCase1)
        {
          Aii[0] -= Aim[0] * *distalAips[0] / *distalAiis[0];
          RHS[0] -= Aim[0] * *distalInputs[0] / *distalAiis[0];
        }
        else if (isDistalCase2)
        {
          // Why do we not adjust Aii[0]? Check.
          RHS[0] -= Aim[0] * *distalInputs[0];
        }
      }
    }
    else{
      Aii[i] = getSharedMembers().bmt - Aim[i] - Aip[i];
    }
    /* * * Sum Currents * * */
    // loop through different kinds of IP3 currents (LCCv12, LCCv13, R-type, ...)
    // 1.a. producing I_IP3 [pA/um^2]
    Array<ChannelIP3Currents>::iterator iter = channelIP3Currents.begin();
    Array<ChannelIP3Currents>::iterator end = channelIP3Currents.end();
    for (; iter != end; iter++)
    {
      RHS[i] -= currentDensityToConc[i] * (*iter->currents)[i];
    }

    // 1.b. producing J_IP3 [uM/msec]
    Array<ChannelIP3Fluxes>::iterator fiter = channelIP3Fluxes.begin();
    Array<ChannelIP3Fluxes>::iterator fend = channelIP3Fluxes.end();
    for (; fiter != fend; fiter++)
    {
      RHS[i] +=  (*fiter->fluxes)[i];
    }
    /* This is a simple implementation of calcium extrusion. To be elaborated as
     * needed. */
    // integrated extrusion mechanism 
    RHS[i] -= IP3Clearance * (IP3_cur[i] - getSharedMembers().IP3Baseline);
    
  }

  //  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<ReceptorIP3Current>::iterator riter = receptorIP3Currents.begin();
  Array<ReceptorIP3Current>::iterator rend = receptorIP3Currents.end();
  for (; riter != rend; riter++)
  {
    int i = riter->index;
    RHS[i] -= currentDensityToConc[i] * *(riter->current);
  }

  // 1.c. HH-like of concentration diffusion
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  Array<TargetAttachIP3Concentration >::iterator ciiter = targetAttachIP3Concentration.begin();
  Array<TargetAttachIP3Concentration >::iterator ciend = targetAttachIP3Concentration.end();
  //dyn_var_t invTime = 1.0/(getSharedMembers().dt;
  for (; ciiter != ciend; ciiter++)
  {
    int i = (ciiter)->index;
    RHS[i] += (*(ciiter->inverseTime)) * (*(ciiter->IP3)); //[uM/ms]
    Aii[i] += (*(ciiter->inverseTime)) ; //[1/ms]
  }
#endif
  
  //  4. injected currents (pA)
  Array<InjectedIP3Current>::iterator iiter = injectedIP3Currents.begin();
  Array<InjectedIP3Current>::iterator iend = injectedIP3Currents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < size)
      RHS[iiter->index] += *(iiter->current) * iiter->currentToConc;
  }

  /* * *  Forward Solve Ax = B * * */
  /* Starting from distal-end (i=0)
   * Eliminate Aim[?] by taking
   * RHS -= Aim[?] * V[proximal]
   * Aii = 
   */
  for (int i = 1; i < size; i++)
  {
    Aii[i] -= Aip[i - 1] * Aim[i] / Aii[i - 1];
    RHS[i] -= RHS[i - 1] * Aim[i] / Aii[i - 1];
  }
}

// Update; IP3_new[] at time (t+dt/2)
// Thomas algorithm backward step 
//   - backward substitution on upper triangular matrix
// Next it calls :finish()
void IP3Concentration::doBackwardSolve()
{
  unsigned size = branchData->size;
  if (isProximalCase0)
  {
    IP3_new[size - 1] = RHS[size - 1] / Aii[size - 1];
  }
  else
  {
    IP3_new[size - 1] =
        (RHS[size - 1] - Aip[size - 1] * *proximalIP3Concentration) /
        Aii[size - 1];
  }
  for (int i = size - 2; i >= 0; i--)
  {
    IP3_new[i] = (RHS[i] - Aip[i] * IP3_new[i + 1]) / Aii[i];
  }
}

//GOAL: get coefficient of Aip or Aim
//  D_IP3 * (r_{i->j})^2 / (dist^2 * b->r^2)
//NOTE: a is the current compartment, and
//      b is the parent compartment (proximal side)
//      index = index of 'a'
dyn_var_t IP3Concentration::getLambda_parent(DimensionStruct* a, 
    DimensionStruct* b,
    int index, 
    bool connectJunction)
{
  dyn_var_t Rb;// radius_middle ()
//#ifdef NEW_DISTANCE_NONUNIFORM_GRID 
//  dyn_var_t dsi = getHalfDistance(index);
//#else
//  dyn_var_t dsi = a->length;
//#endif
  dyn_var_t distance;
  dyn_var_t volume = getVolume(index);
  if (a->dist2soma <= SMALL)//avoid the big soma
  {//a  CAN't BE the compartment representing 'soma'
    assert(0);
  }
  else if (b->dist2soma <= SMALL)
  {//b is the compartment representing 'soma'
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA_WITH;
    //END TEST
#endif

#ifdef USE_SOMA_AS_ISOPOTENTIAL
    distance = std::fabs(a->dist2soma - b->r); // SOMA is treated as a point source
#else
    distance = std::fabs(a->dist2soma - b->dist2soma);
#ifdef USE_STRETCH_SOMA_RADIUS
    //TEST 
    distance += STRETCH_SOMA_WITH;
    //  distance += 50.0;//TUAN TESTING - make soma longer
    //distance = std::fabs(b->r + a->dist2soma);
    //END TEST
#endif
#endif
  }
  else
  {
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
    if (connectJunction)
      Rb = b->r;
    else
      Rb = 0.5 * (a->r + b->r);
#else
    Rb = 0.5 * (a->r + b->r);
#endif
    distance = std::fabs(b->dist2soma - a->dist2soma);
  }
  return (M_PI * Rb * Rb * D_IP3) / 
      (volume * distance);
}
//GOAL: get coefficient of Aip or Aim
//  D_IP3 * (r_{i->j})^2 / (dist^2 * b->r^2)
//NOTE: a is the current compartment, and
//      b is the child compartment (distal side)
//      index = index of 'a'
dyn_var_t IP3Concentration::getLambda_child(DimensionStruct* a, 
    DimensionStruct* b,
    int index, 
    bool connectJunction)
{
  dyn_var_t Rb;// radius_middle ()
//#ifdef NEW_DISTANCE_NONUNIFORM_GRID 
//  dyn_var_t dsi = getHalfDistance(index);
//#else
//  dyn_var_t dsi = a->length;
//#endif
  dyn_var_t distance;
  dyn_var_t volume = getVolume(index);
  if (a->dist2soma <= SMALL)//avoid the big soma
  {//a  CAN't BE the compartment representing 'soma'
    assert(0);
  }
  else if (b->dist2soma <= SMALL)
  {//b is the compartment representing 'soma'
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA_WITH;
    //END TEST
#endif

#ifdef USE_SOMA_AS_ISOPOTENTIAL
    distance = std::fabs(a->dist2soma - b->r); // SOMA is treated as a point source
#else
    distance = std::fabs(a->dist2soma - b->dist2soma);
#ifdef USE_STRETCH_SOMA_RADIUS
    //TEST 
    distance += STRETCH_SOMA_WITH;
    //  distance += 50.0;//TUAN TESTING - make soma longer
    //distance = std::fabs(b->r + a->dist2soma);
    //END TEST
#endif
#endif
  }
  else
  {
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
    if (connectJunction)
      Rb = b->r;
    else
      Rb = 0.5 * (a->r + b->r);
#else
    Rb = 0.5 * (a->r + b->r);
#endif
    distance = std::fabs(b->dist2soma - a->dist2soma);
  }
  return (M_PI * Rb * Rb * D_IP3) / 
      (volume * distance);
}
//GOAL: get coefficient of Aip or Aim
//  D_IP3 * (r_{i->j})^2 / (dist^2 * b->r^2)
//NOTE: a is the current compartment, and
//      b is the adjacent compartment (can be proximal or distal side)
dyn_var_t IP3Concentration::getLambda(DimensionStruct* a, 
    DimensionStruct* b,
    int index, 
    bool connectJunction)
{
  dyn_var_t Rb;// radius_middle ()
#ifdef NEW_DISTANCE_NONUNIFORM_GRID 
  dyn_var_t dsi = getHalfDistance(index);
#else
  dyn_var_t dsi = a->length;
#endif
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)//avoid the big soma
  {//a  CAN't BE the compartment representing 'soma'
    assert(0);
  }
  else if (b->dist2soma <= SMALL)
  {//b is the compartment representing 'soma'
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA;
    //END TEST
#endif

#ifdef USE_SOMA_AS_ISOPOTENTIAL
    distance = std::fabs(a->dist2soma - b->r); // SOMA is treated as a point source
#else
    distance = std::fabs(a->dist2soma);
#ifdef USE_STRETCH_SOMA_RADIUS
    //TEST 
    distance += STRETCH_SOMA_WITH;
    //  distance += 50.0;//TUAN TESTING - make soma longer
    //distance = std::fabs(b->r + a->dist2soma);
    //END TEST
#endif
#endif
  }
  else
  {
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
    if (connectJunction)
      Rb = b->r;
    else
      Rb = 0.5 * (a->r + b->r);
#else
		Rb = 0.5 * (a->r + b->r);
#endif
    distance = std::fabs(b->dist2soma - a->dist2soma);
  }
  return (D_IP3 * Rb * Rb /
          (dsi * distance * a->r * a->r)); /* needs fixing */
  /* NOTE: ideally
  return (D_IP3  /
          (dsi * distance )); 
          */
}
//find the lambda between the terminal point of the 
//compartment represented by 'a'
//'a' can be cpt[0] (distal-end) or cpt[size-1] (proximal-end)
dyn_var_t IP3Concentration::getLambda(DimensionStruct* a, int index)
{
  dyn_var_t Rb ;// radius_middle ()
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
  {//a  CAN't BE the compartment representing 'soma'
    assert(0);
  }
  else
  {
    Rb = a->r;
    distance = std::fabs(a->length/2.0);
  }
#ifdef NEW_DISTANCE_NONUNIFORM_GRID //if defined, then ensure 
  dyn_var_t dsi ;
  if (index == 0)
    dsi = (a->length/2.0 + std::fabs(a->dist2soma - dimensions[1]->dist2soma));
  else if (index == branchData->size-1)
    dsi = (a->length/2.0 + std::fabs(a->dist2soma - dimensions[index-2]->dist2soma));
  else
    assert(0);
#else
  dyn_var_t dsi  = distance;
#endif
  return (D_IP3 * Rb * Rb /
          (dsi * distance * a->r * a->r)); /* needs fixing */
  /* NOTE: ideally
  return (D_IP3  /
          (dsi * distance )); 
          */
}

// GOAL: Get coefficient of Aip[0] and Aim[size-1]
//  for Cai(i=0,j=branch-index)
// i.e. at implicit branch point
//  D_IP3 * (1/V) * PI * r_(i->j)^2 / (ds_(i->j))
//   V = volume of cytosolic compartment
//   D_IP3 = diffusion constant of Ca(cyto)
//  NOTE: 'a' is the current node;
//        'b' is the proxomal-side node 
dyn_var_t IP3Concentration::getAij_parent(DimensionStruct* a, DimensionStruct* b,
    dyn_var_t V, bool connectJunction)
{
  dyn_var_t Rb;
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
  {
    assert(0); // a CANNOT be soma
  }
  else if (b->dist2soma <= SMALL)
  {
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA_WITH;
    //END TEST
#endif
#ifdef USE_SOMA_AS_ISOPOTENTIAL
    distance = std::fabs(a->dist2soma - b->r); // SOMA is treated as a point source
#else
    //distance = fabs(b->r + a->dist2soma );
    distance = std::fabs(a->dist2soma);
#ifdef USE_STRETCH_SOMA_RADIUS
    //TEST 
    distance += STRETCH_SOMA_WITH;
    //  distance += 50.0;//TUAN TESTING - make soma longer
    //distance = std::fabs(b->r + a->dist2soma);
    //END TEST
#endif
#endif
  }
  else
  {
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
    if (connectJunction)
      Rb = b->r;
    else
      Rb = 0.5 * (a->r + b->r);
#else
    Rb = 0.5 * (a->r + b->r);
#endif
    distance = fabs(b->dist2soma - a->dist2soma);
  }
  return (M_PI * Rb * Rb * D_IP3 /
      (V * distance));
}

// GOAL: Get coefficient of Aip[0] and Aim[size-1]
//  for Cai(i=0,j=branch-index)
// i.e. at implicit branch point
//  D_IP3 * (1/V) * PI * r_(i->j)^2 / (ds_(i->j))
//   V = volume of cytosolic compartment
//   D_IP3 = diffusion constant of Ca(cyto)
//  NOTE: 'a' is the current node;
//        'b' is the distal-side node 
dyn_var_t IP3Concentration::getAij_child(DimensionStruct* a, DimensionStruct* b,
    dyn_var_t V, bool connectJunction)
{
  dyn_var_t Rb;
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
  {
    assert(0); // a CANNOT be soma
  }
  else if (b->dist2soma <= SMALL)
  {
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA_WITH;
    //END TEST
#endif
#ifdef USE_SOMA_AS_ISOPOTENTIAL
    distance = std::fabs(a->dist2soma - b->r); // SOMA is treated as a point source
#else
    //distance = fabs(b->r + a->dist2soma );
    distance = std::fabs(a->dist2soma);
#ifdef USE_STRETCH_SOMA_RADIUS
    //TEST 
    distance += STRETCH_SOMA_WITH;
    //  distance += 50.0;//TUAN TESTING - make soma longer
    //distance = std::fabs(b->r + a->dist2soma);
    //END TEST
#endif
#endif
  }
  else
  {
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
    if (connectJunction)
      Rb = b->r;
    else
      Rb = 0.5 * (a->r + b->r);
#else
    Rb = 0.5 * (a->r + b->r);
#endif
    distance = fabs(b->dist2soma - a->dist2soma);
  }
  return (M_PI * Rb * Rb * D_IP3 /
      (V * distance));
}

// GOAL: Get coefficient of Aip[0] and Aim[size-1]
//  for IP3(i=0,j=branch-index)
// i.e. at implicit branch point
//  D_IP3 * (1/V) * PI * r_(i->j)^2 / (ds_(i->j))
//   V = volume of cytosolic compartment
//   D_IP3 = diffusion constant of IP3(cyto)
//  NOTE: 'a' is the distal-end compartment of the branch (i=0)
//        serving as implicit branch 
dyn_var_t IP3Concentration::getAij(DimensionStruct* a, DimensionStruct* b,
                                  dyn_var_t V, bool connectJunction)
{
  dyn_var_t Rb;
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
  {
    assert(0); // a CANNOT be soma
  }
  else if (b->dist2soma <= SMALL)
  {
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA_WITH;
    //END TEST
#endif
#ifdef USE_SOMA_AS_ISOPOTENTIAL
    distance = std::fabs(a->dist2soma - b->r); // SOMA is treated as a point source
#else
    //distance = fabs(b->r + a->dist2soma );
    distance = std::fabs(a->dist2soma);
#ifdef USE_STRETCH_SOMA_RADIUS
    //TEST 
    distance += STRETCH_SOMA_WITH;
    //  distance += 50.0;//TUAN TESTING - make soma longer
    //distance = std::fabs(b->r + a->dist2soma);
    //END TEST
#endif
#endif
  }
  else
  {
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
    if (connectJunction)
      Rb = b->r;
    else
      Rb = 0.5 * (a->r + b->r);
#else
    Rb = 0.5 * (a->r + b->r);
#endif
    distance = fabs(b->dist2soma - a->dist2soma);
  }
  return (M_PI * Rb * Rb * D_IP3 /
      (V * distance));
}


void IP3Concentration::setReceptorIP3Current(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
    CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(receptorIP3Currents.size() > 0);
#endif
  receptorIP3Currents[receptorIP3Currents.size() - 1].index = CG_inAttrPset->idx;
}

// to be called at connection-setup time
//    check MDL for what kind of connection then it is called
void IP3Concentration::setInjectedIP3Current(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset,
    CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(injectedIP3Currents.size() > 0);
#endif
  TissueSite& site = CG_inAttrPset->site;
  if (site.r != 0)  // a sphere is provided, i.e. used for current injection
  {//stimulate a region (any compartments fall within the sphere are affected)
    // go through all compartments
    for (int i = 0; i < dimensions.size(); ++i)
    {
      //.. check the distance between that compartment and the size
      //   here if it falls inside the sphere then connection established
      //     for bidirectional connection
      if ((site.r * site.r) >= DISTANCE_SQUARED(&site, dimensions[i]))
      {
        IP3CurrentProducer* CG_IP3CurrentProducerPtr =
            dynamic_cast<IP3CurrentProducer*>(CG_variable);
        if (CG_IP3CurrentProducerPtr == 0)
        {
          std::cerr
              << "Dynamic Cast of CurrentProducer failed in IP3Concentration"
              << std::endl;
          exit(-1);
        }
        injectedIP3Currents.increase();
        injectedIP3Currents[injectedIP3Currents.size() - 1].current =
            CG_IP3CurrentProducerPtr->CG_get_IP3CurrentProducer_current();
        injectedIP3Currents[injectedIP3Currents.size() - 1].index = i;
        checkAndAddPreVariable(CG_variable);
      }
    }
  }
  else if (CG_inAttrPset->idx < 0)  // Can be used via 'Probe' of TissueFunctor
  {//inject at all compartments of one or many branchs meet the condition
    injectedIP3Currents[injectedIP3Currents.size() - 1].index = 0;
    for (int i = 1; i < branchData->size; ++i)
    {
      IP3CurrentProducer* CG_IP3CurrentProducerPtr =
          dynamic_cast<IP3CurrentProducer*>(CG_variable);
      if (CG_IP3CurrentProducerPtr == 0)
      {
        std::cerr
          << "Dynamic Cast of CurrentProducer failed in IP3Concentration"
          << std::endl;
        exit(-1);
      }
      injectedIP3Currents.increase();
      injectedIP3Currents[injectedIP3Currents.size() - 1].current =
          CG_IP3CurrentProducerPtr->CG_get_IP3CurrentProducer_current();
      injectedIP3Currents[injectedIP3Currents.size() - 1].index = i;
      checkAndAddPreVariable(CG_variable);
    }
  }
  else
  {//i.e. bi-directional connection (electrical synapse or spineneck-compartment)
   //NOTE: The current component already been assigned via code-generated specified in MDL
    injectedIP3Currents[injectedIP3Currents.size() - 1].index =
        CG_inAttrPset->idx;
  }
}

IP3Concentration::~IP3Concentration() {}

dyn_var_t IP3Concentration::getHalfDistance (int index) 
{
  dyn_var_t halfDist = 0.0 ;
  unsigned size = branchData->size;  //# of compartments
  assert(index >=0 and index <= size-1);
  if  (index == size-1)
  {
    if (! isProximalCase0)
    {
      if (proximalDimension->dist2soma <= SMALL)
      {
        if (size==1)
        {
          if (isDistalCase0)
          {//no flux distal
            halfDist = ( dimensions[index]->length/2 );
          }
          else if (isDistalCase1 or isDistalCase2)
          {
            halfDist = (
                std::fabs( dimensions[index]->length/2 )
                +
                std::fabs( dimensions[index]->dist2soma - distalDimensions[0]->dist2soma )
                )/ 2.0;
          }
        }
        else
        {
          halfDist = (
              std::fabs( dimensions[index]->length/2 )
              +
              std::fabs( dimensions[index]->dist2soma - dimensions[index-1]->dist2soma )
              )/ 2.0;
          //halfDist = (
          //    std::fabs( dimensions[index]->dist2soma - dimensions[index-1]->dist2soma )
          //    );

        }
      }
      else{
        if (size==1)
        {
          if (isDistalCase0)
            halfDist = (
                std::fabs( dimensions[index]->dist2soma - proximalDimension->dist2soma )
                );
          else if (isDistalCase1 or isDistalCase2)
            halfDist = (
                std::fabs( dimensions[index]->dist2soma - proximalDimension->dist2soma )
                +
                std::fabs( dimensions[index]->dist2soma - distalDimensions[0]->dist2soma )
                )/ 2.0;
        }
        else
          halfDist = (
              std::fabs( dimensions[index]->dist2soma - proximalDimension->dist2soma )
              +
              std::fabs( dimensions[index]->dist2soma - dimensions[index-1]->dist2soma )
              )/ 2.0;
      }

    }
    else
      halfDist = (
          std::fabs( dimensions[index]->dist2soma - dimensions[index-1]->dist2soma )
          );
  }
  else if (index == 0)
    if (isDistalCase0)
      halfDist = (
          std::fabs( dimensions[index]->dist2soma - dimensions[index+1]->dist2soma )
          );
    else if (isDistalCase1 or isDistalCase2)
      halfDist = (
          std::fabs( dimensions[index]->dist2soma - distalDimensions[0]->dist2soma )
          +
          std::fabs( dimensions[index+1]->dist2soma - dimensions[index]->dist2soma )
          )/ 2.0;
    else 
    {// no use
    }
  else 
  {
    halfDist = (
        std::fabs( dimensions[index]->dist2soma - dimensions[index-1]->dist2soma )
        +
        std::fabs( dimensions[index+1]->dist2soma - dimensions[index]->dist2soma )
        )/ 2.0;
  }
  return halfDist;
}



#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
void IP3Concentration::setTargetAttachIP3Concentration(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IP3ConcentrationInAttrPSet* CG_inAttrPset, CG_IP3ConcentrationOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(targetAttachIP3Concentration.size() > 0);
#endif
  targetAttachIP3Concentration[targetAttachIP3Concentration.size() - 1].index = CG_inAttrPset->idx;

}

#endif
