// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
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
#include "MaxComputeOrder.h"
#include "GlobalNTSConfig.h"

#include <iomanip>
#include <cmath>

#define TEST_AIJ
#define TEST_LAMBDA //  working now - must use surface area 
#define SECOND_ORDER_SPATIAL // second-order accuracy in spatial is achived when (2, -2) coefficient is used
                             // for von Neuman condition

#define SMALL 1.0E-6
#define DISTANCE_SQUARED(a, b)               \
  ((((a)->x - (b)->x) * ((a)->x - (b)->x)) + \
   (((a)->y - (b)->y) * ((a)->y - (b)->y)) + \
   (((a)->z - (b)->z) * ((a)->z - (b)->z)))

SegmentDescriptor HodgkinHuxleyVoltage::_segmentDescriptor;

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

// TUAN: now try to revise the code so that it can be adopted to Ca, CaER easily

//#define DEBUG_HH
// Conserved region (only change ClassName)
//{{{
void HodgkinHuxleyVoltage::solve(RNG& rng)
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
void HodgkinHuxleyVoltage::forwardSolve1(RNG& rng)
{
  if (computeOrder == 1)
  {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve1(RNG& rng)
{
  if (computeOrder == 1) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 1
void HodgkinHuxleyVoltage::forwardSolve2(RNG& rng)
{
  if (computeOrder == 2)
  {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve2(RNG& rng)
{
  if (computeOrder == 2) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 2
void HodgkinHuxleyVoltage::forwardSolve3(RNG& rng)
{
  if (computeOrder == 3)
  {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve3(RNG& rng)
{
  if (computeOrder == 3) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 3
void HodgkinHuxleyVoltage::forwardSolve4(RNG& rng)
{
  if (computeOrder == 4)
  {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve4(RNG& rng)
{
  if (computeOrder == 4) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 4
void HodgkinHuxleyVoltage::forwardSolve5(RNG& rng)
{
  if (computeOrder == 5)
  {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve5(RNG& rng)
{
  if (computeOrder == 5) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 5
void HodgkinHuxleyVoltage::forwardSolve6(RNG& rng)
{
  if (computeOrder == 6)
  {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve6(RNG& rng)
{
  if (computeOrder == 6) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 6
void HodgkinHuxleyVoltage::forwardSolve7(RNG& rng)
{
  if (computeOrder == 7)
  {
    doForwardSolve();
  }
}

void HodgkinHuxleyVoltage::backwardSolve7(RNG& rng)
{
  if (computeOrder == 7) doBackwardSolve();
}
#endif

bool HodgkinHuxleyVoltage::confirmUniqueDeltaT(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset,
    CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}

//TUAN: TODO challenge
//   how to check for 2 sites overlapping
//   if we don't retain the dimension's (x,y,z) coordinate
//  Even if we retain (x,y,z) this value change with the #capsule per compartment
//   and geometric sampling --> so not a good choice
bool HodgkinHuxleyVoltage::checkSite(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset,
    CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
  TissueSite& site = CG_inAttrPset->site;
  bool atSite = (site.r == 0);
  for (unsigned int i = 0; !atSite && i < dimensions.size(); ++i)
    atSite = ((site.r * site.r) >= DISTANCE_SQUARED(&site, dimensions[i]));
  return atSite;
}

void HodgkinHuxleyVoltage::setProximalJunction(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset,
    CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
  proximalJunction = true;
}

// update: V(t+dt) = 2 * V(t+dt/2) - V(t) = V(t) + 2 * DeltaV
//  This second step is basically a forward Euler method with time-step dt/2
// second-step (final step) in Crank-Nicholson method (secondorder=2)
//  REMIND: The Crank-Nicholson method is 2 stage: 
//       backward Euler for t -> t+dt/2
//       forward Euler for   t+dt/2 -> t+dt
void HodgkinHuxleyVoltage::finish(RNG& rng)
{
  //TUAN DEBUG
#ifdef DEBUG_COMPARTMENT
  volatile unsigned nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile unsigned bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile unsigned iteration = getSimulation().getIteration();
#endif
  //END TUAN DEBUG
  unsigned size = branchData->size;
#ifdef DEBUG_HH
  printDebugHH();
#endif
  for (int i = 0; i < size; ++i)
  {
    Vcur[i] = Vnew[i] = 2.0 * Vnew[i] - Vcur[i];
#ifdef DEBUG_ASSERT
    if (not (Vnew[i] == Vnew[i])
        //			or std::fabs(Vnew[i]-Vold)/(*getSharedMembers().deltaT)
        or Vnew[i]> 230.0
        or Vnew[i] < -330.0
       )
    {
      printDebugHH();
      std::cerr << "compartment " << i << "-th\n";
      assert(0);
      assert(Vnew[i] == Vnew[i]);  // making sure Vnew[i] is not NaN
    }
#endif
  }
}

// Get membrane surface area of the compartment based on its index 'i'
inline dyn_var_t HodgkinHuxleyVoltage::getArea(int i)  // Tuan: check ok
{
#ifdef DEBUG_ASSERT
  assert(i >= 0 && i < branchData->size);
#endif
  dyn_var_t area = dimensions[i]->surface_area;
  return area;
}
//}}} //end Conserved region

// GOAL: initialize data at each branch
//    the compartments along one branch are indexed from distal (index=0)
//    to the proximal (index=branchData->size-1)
//    so Aim[..] from distal side
//       Aip[..] from proximal side
void HodgkinHuxleyVoltage::initializeCompartmentData(RNG& rng) 
{
  //NOTE: 
  // 2 things to be updated
  //   For current induced by microelectrode impalement --> shunt is restricted to a small enough region
  //      So it is considered as a POINT process and is described as total current (or total resistance)
  //      and all of its current flows at one site (rather than equally distributed over an area)
  //      I = (Vc - Vm) / r
  //   Synaptic current is also considered as a POINT PROCESS
  //   Injected currents are treated as POINT PROCESS --> injected with level Iinj directly to the point
  //      e.g. shunt current Ishunt = (V)
  //
  // Secondly: page 168-189 in NEURON book
  //   Suppose current node is 'i' and adjacent node being considered is 'j'
  //   If j refers parent node of 'i': use the area of 'i'-surface area
  //   If j is a child-node of 'i': use the area of the child's node
  // The Aip[] should get the proximal due to the way V[..] array is indexed (0=distal, nCpts-1=proximal)
  //TUAN DEBUG
#ifdef DEBUG_COMPARTMENT
  volatile unsigned nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile unsigned bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile unsigned iteration = getSimulation().getIteration();
#endif
  //END TUAN DEBUG
  // for a given computing process:
  //  here all the data in vector-form are initialized to
  //  the same size as the number of compartments in a branch (i.e. branchData)
  unsigned numCpts = branchData->size;  //# of compartments
  computeOrder = _segmentDescriptor.getComputeOrder(branchData->key);

  if (isProximalCase2) 
  {
    if (computeOrder != 0)
      printDebugHH();
    assert(computeOrder == 0);
  } 
  if (isDistalCase2) assert(computeOrder == MAX_COMPUTE_ORDER);
  assert(dimensions.size() == numCpts);
  assert(Vnew.size() == numCpts);
  assert(distalDimensions.size() == distalInputs.size());


  // allocate data
  if (Vcur.size() != numCpts) Vcur.increaseSizeTo(numCpts);
  if (Aii.size() != numCpts) Aii.increaseSizeTo(numCpts);
  if (Aip.size() != numCpts) Aip.increaseSizeTo(numCpts);
  if (Aim.size() != numCpts) Aim.increaseSizeTo(numCpts);
  if (RHS.size() != numCpts) RHS.increaseSizeTo(numCpts);

  // initialize data
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
      Vcur[i] = Vm_values[j];
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
  Vcur[i] = Vm_values[j-1];
      }
      else if (j < Vm_values.size()) 
  Vcur[i] = Vm_values[j];
      else
  Vcur[i] = Vm_default;
    }
    else {
      Vcur[i] = Vm_default;
    }
  }
  for (int i = 1; i < numCpts; ++i)
  {
    Vnew[i] = Vcur[i];
  }
#else
  Vcur[0] = Vnew[0];
  for (int i = 1; i < numCpts; ++i)
  {
    Vnew[i] = Vnew[0];
    Vcur[i] = Vcur[0];
  }
#endif

  // go through each compartments in a branch
  for (int i = 0; i < numCpts; ++i)
  {//reset
    Aii[i] = Aip[i] = Aim[i] = RHS[i] = 0.0;
  }

  // get surface area of the compartment and put into InjectedCurrent structure
  Array<InjectedCurrent>::iterator iiter = injectedCurrents.begin();
  Array<InjectedCurrent>::iterator iend = injectedCurrents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < numCpts) iiter->area = getArea(iiter->index);
  }

  //Aim[0] = Aip[numCpts - 1] = 0;
  if (!isProximalCase0)
  {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
    //if (isProximalCase1)
    //{
    //  Aip[numCpts - 1] =
    //    -getLambda(dimensions[numCpts - 1], numCpts-1);  // [nS/um^2]
    //}
    //else{
    //  Aip[numCpts - 1] =
    //    -getLambda(dimensions[numCpts - 1], proximalDimension, numCpts-1, true);  // [nS/um^2]
    //}
    assert(0);
#else
#ifdef TEST_LAMBDA
    Aip[numCpts - 1] =
        -getLambda_parent(dimensions[numCpts - 1], proximalDimension, numCpts-1, true);  // [nS/um^2]
#else
    Aip[numCpts - 1] =
        -getLambda(dimensions[numCpts - 1], proximalDimension, numCpts-1, true);  // [nS/um^2]
#endif
#endif
  }

  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (isDistalCase1 || isDistalCase2)
  {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
    //if (isDistalCase1)
    //  Aim[0] = -getLambda(dimensions[0], 0);
    //else
    //  Aim[0] = -getLambda_child(dimensions[0],distalDimensions[0], 0, true);
    assert(0);
#else
#ifdef TEST_LAMBDA
    Aim[0] = -getLambda_child(dimensions[0],distalDimensions[0], 0, true);
#else
    Aim[0] = -getLambda(dimensions[0],distalDimensions[0], 0, true);
#endif
#endif
  }

  for (int i = 1; i < numCpts; i++)
  {
#ifdef TEST_LAMBDA
    Aim[i] = -getLambda_child(dimensions[i], dimensions[i - 1], i);
#else
    Aim[i] = -getLambda(dimensions[i], dimensions[i - 1], i);
#endif
  }

  for (int i = 0; i < numCpts - 1; i++)
  {
#ifdef TEST_LAMBDA
    Aip[i] = -getLambda_parent(dimensions[i], dimensions[i + 1], i);
#else
    Aip[i] = -getLambda(dimensions[i], dimensions[i + 1], i);
#endif
  }

  if (isDistalCase3)
  {
    // Compute total area of the junction...
    dyn_var_t area = getArea(0);

    // Compute Aij[n] for the junction...one of which goes in Aip[0]...
    if (numCpts == 1)
    {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
      ////CHECK AGAIN
      //if (isProximalCase1)
      //{
      //  Aip[0] = -getLambda(dimensions[0], i);  // [nS/um^2]
      //}
      //else{
      //  Aip[0] = -getAij_parent(dimensions[0], proximalDimension, area, true);
      //}
      assert(0);
#else
#ifdef TEST_AIJ
      Aip[0] = -getAij_parent(dimensions[0], proximalDimension, area, true);
#else
      Aip[0] = -getAij(dimensions[0], proximalDimension, area, true);
#endif
#endif
    }
    else
    {
#ifdef TEST_AIJ
      Aip[0] = -getAij_parent(dimensions[0], dimensions[1], area);
#else
      Aip[0] = -getAij(dimensions[0], dimensions[1], area);
#endif
    }
    /* We revert to the original approach,  
  //IMPORTANT CHANGE: new approach
  // Unlike the original approach
  //   which doesn't have a compartment for the implicit branching
  // The branch now has
  //at least 2: one compartment as implicit branching point + one as regular
  //    compartment-zero as implicit branching compartment
  //    compartment-1th and above as normal
      Aip[0] = -getAij(dimensions[1], dimensions[0], area);
  */
    for (int n = 0; n < distalDimensions.size(); n++)
    {//to be used in place of Aim[]
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
      assert(0);
      //CHECK AGAIN
      Aij.push_back(-getAij_child(dimensions[0], distalDimensions[n], area, true));
#else
#ifdef TEST_AIJ
      Aij.push_back(-getAij_child(dimensions[0], distalDimensions[n], area, true));
#else
      Aij.push_back(-getAij(dimensions[0], distalDimensions[n], area, true));
#endif
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


  if (getSharedMembers().deltaT)
  {
    cmt = 2.0 * Cm / *(getSharedMembers().deltaT);  // [pF/(um^2 . ms)]
  }
#ifdef DEBUG_HH
  printDebugHH();
#endif
}

void HodgkinHuxleyVoltage::printDebugHH()
{
  unsigned size = branchData->size;
  for (int i = 0; i < size; ++i)
  {
    this->printDebugHH(i);
  }
}

void HodgkinHuxleyVoltage::printDebugHH(int cptIndex)
{
  unsigned size = branchData->size;
  if (cptIndex == 0)
  {
    std::cerr << "iter,time| BRANCH [rank, nodeIdx, layerIdx, cptIdx]"
      << "(neuronIdx, brIdx, brOrder, brType) | distal(C0 | C1 | C2 | C3) :"
      << " prox(C0 | C1 | C2) |"
      << "{x,y,z,r | dist2soma, surface_area, volume, length} Vm\n";
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
    << Vnew[i]  << " " << std::endl;
}

void HodgkinHuxleyVoltage::printDebugHHCurrent(int cptIndex)
{
  unsigned size = branchData->size;
  int i  = cptIndex;
  Array<InjectedCurrent>::iterator iiter = injectedCurrents.begin();
  Array<InjectedCurrent>::iterator iend = injectedCurrents.end();
  std::cerr << "==================" << std::endl;
  for (; iiter != iend; iiter++)
  {
    if (iiter->index == cptIndex) 
    {
      std::cerr << "Inj [pA/um^2] = " << *(iiter->current) / iiter->area //; // [pA/um^2]
        << " [pA] = " << *(iiter->current) << ", [um^2] = " << iiter->area
          << " | " << getArea(iiter->index) << std::endl;
    }
  }
  std::cerr << "==================" << std::endl;
}
// NOTE: Vnew = Vcur = Vm(t) at time (t = tentry)
// GOAL Recalculate: RHS[], Aii[] at time (t+dt/2)
// Unit: RHS = current density (pA/um^2)
//       Aii = conductance density (nS/um^2) data at time (t+dt/2)
// Convert to upper triangular matrix  
// Thomas algorithm forward step 
//  /* * *  Forward Solve Ax = B * * */
//  /* Starting from distal-end (i=0)
//   * Eliminate Aim[?] by taking
//   * RHS -= Aim[?] * V[proximal]
//   * Aii = 
//   */
// 
void HodgkinHuxleyVoltage::doForwardSolve()
{
  //TUAN DEBUG
#ifdef DEBUG_COMPARTMENT
  volatile unsigned nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile unsigned bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile unsigned iteration = getSimulation().getIteration();
#endif
  //END TUAN DEBUG
  unsigned size = branchData->size;
  //Find A[ii]i and RHS[ii]  
  //  1. ionic currents 
  for (int i = 0; i < size; i++)  // for each compartment on that branch
  {
    RHS[i] = cmt * Vnew[i] + gLeak * getSharedMembers().E_leak;   //[[pA/um^2]]
    if (i == 0)
    {
      if ((isDistalCase3))
      {//the node at index 0 is implicit junction
        // series of Aij[..] is used in place of Aim[0]
        Aii[i] = cmt + gLeak - Aip[i];    // [nS/um^2]
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
        ////no change to Aii[0]
        /*
        //for (int n = 0; n < distalInputs.size(); n++)
        //{
        //  Aii[0] -= Aij[n];
        //  RHS[0] -= Aij[n] * *distalInputs[n];
        //}
        */
        assert(0);
#else
        for (int n = 0; n < distalInputs.size(); n++)
        {
          Aii[0] -= Aij[n];
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
          assert(0);
          Aii[0] -= Aij[n] ;
          RHS[0] -= Aij[n] * *distalInputs[n] ;
          RHS[0] /= Aii[0];
          Aip[0] /= Aii[0];
#else
          Aii[0] -= Aij[n] * *distalAips[n] / *distalAiis[n];
          RHS[0] -= Aij[n] * *distalInputs[n] / *distalAiis[n];
#endif
        }
#endif
      }
      else{
        Aii[i] = cmt + gLeak - Aim[i] - Aip[i];    // [nS/um^2]
        if (isDistalCase1) {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
          assert(0);
          //Aii[0] = 0; //-= Aim[0] * *distalAips[0] / *distalAiis[0];
          dyn_var_t V1;
          dyn_var_t w1 = 1.0/(distalDimensions[0]->length);
          dyn_var_t w2 = 1.0/(dimensions[0]->length);
          V1 = (w1 * *distalInputs[0] + w2 * Vnew[0])/(w1+w2);
          //no change Aii[0]
          RHS[0] -= Aim[0] * V1; 
          RHS[0] /= Aii[0];
          Aip[0] /= Aii[0];
#else
          //NOTE: distalAiis ~ conductance
          //      distalAips ~ conductance
          Aii[0] -= Aim[0] * *distalAips[0] / *distalAiis[0];
          RHS[0] -= Aim[0] * *distalInputs[0] / *distalAiis[0];
#endif
        }
        else if (isDistalCase2)
        {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
          assert(0);
          RHS[0] -= Aim[0] * *distalInputs[0];
          RHS[0] /= Aii[0];
          Aip[0] /= Aii[0];
#else
          // not adjust Aii[0] - as the Jacobian matrix end at explicit junction
          RHS[0] -= Aim[0] * *distalInputs[0];
#endif
        }
      }
    }
    else{
        Aii[i] = cmt + gLeak - Aim[i] - Aip[i];    // [nS/um^2]
    }
    //QUESTION: why not subtracting 
    //     Aip[0]*proximalVoltage; // in RHS[0] here
    //     Aip[i]*Vnew[i+1]; // in RHS[i] here
    //ANSWER: this step convert to upper-triangular matrix
    // so elimination Aip[] is done in :doBackwardSolve()
    
    /* * * Sum Currents * * */
    // loop through different kinds of currents (Kv, Nav1.6, ...)
    //  1.a. ionic currents using Hodgkin-Huxley type equations (+g*Erev)
    Array<ChannelCurrents>::iterator iter = channelCurrents.begin();
    Array<ChannelCurrents>::iterator end = channelCurrents.end();
    for (int k = 0; iter != end; iter++, ++k)
    {
      //NOTE: the conductance is already second-order accuracy at (tentry+dt/2)
      ShallowArray<dyn_var_t>* conductances = iter->conductances;
      // at each current type, there is an array of currents of that type,
      //...each current element flows into one compartment
      //BUG: As reversalPotential is on Shared, they are assumed the same for all ChannelNat
      // so we cannot get the index based on the compartment-index
      /*
      RHS[i] +=
          (*conductances)[i] *
          (*(iter->reversalPotentials))[(iter->reversalPotentials->size() == 1)
                                            ? 0
                                            : i];
      */
      // Fixed - for now
      RHS[i] +=
          (*conductances)[i] *
          (*(iter->reversalPotentials))[0];

      Aii[i] += (*conductances)[i];
    }

    //  1.b. ionic currents using GHK equations (-Iion)
    Array<ChannelCurrentsGHK>::iterator iiter = channelCurrentsGHK.begin();
    Array<ChannelCurrentsGHK>::iterator iend = channelCurrentsGHK.end();
    for (; iiter != iend; iiter++)
    {
      //IMPORTANT: subtraction is used
      RHS[i] -=  (*(iiter->currents))[i]; //[pA/um^2]
#ifdef CONSIDER_DI_DV
      //take into account di/dv * Delta_V
      //IMPORTANT: addition is used
      ////TODO IMPORTANT
      //RHS[i] += di_dv * Vcur[i]; 
      //Aii[i] += di_dv;  
      RHS[i] +=  (*(iiter->di_dv))[i] * Vcur[i]; //[pA/um^2]
      Aii[i] +=  (*(iiter->di_dv))[i]; //[pA/um^2]
#endif
    }
  }

  //  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<ReceptorCurrent>::iterator riter = receptorCurrents.begin();
  Array<ReceptorCurrent>::iterator rend = receptorCurrents.end();
  for (; riter != rend; riter++)
  {
    int i = riter->index;
    RHS[i] += *(riter->conductance) * *(riter->reversalPotential);
    Aii[i] += *(riter->conductance);
  }

  //  3. receptor currents using GHK type equations (gV, gErev)
  //  TUAN : TODO consider if this happens
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

  //  4. injected currents
  /* Note: Injected currents comprise current interfaces produced by two
     different
     categories of models: 1) experimentally injected currents, as in a patch
     clamp
     electrode in current clamp mode, and 2) electrical synapse currents, as in
     the
     current injected from one compartment to another via a gap junction.

     Since we think of injected currents as positive quantities with units of
     pA,
     the sign on injected currents is reversed, and the units are pA and not
     pA/um^2,
     even for electrical synapses.
  */
  Array<InjectedCurrent>::iterator iiter = injectedCurrents.begin();
  Array<InjectedCurrent>::iterator iend = injectedCurrents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < branchData->size)
      RHS[iiter->index] += *(iiter->current) / iiter->area; // [pA/um^2]
  }

  //apply Gaussian elimination to remove Aim[] factors
  // distal toward proximal
  for (int i = 1; i < size; i++)
  {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
    assert(0);
    Aip[i] /= (Aii[i] - Aim[i] * Aip[i - 1] );
    RHS[i] = (RHS[i] - Aim[i] * RHS[i-1]) / (Aii[i] - Aim[i] * Aip[i - 1]);
#else
    Aii[i] -= Aip[i - 1] * Aim[i] / Aii[i - 1];
    RHS[i] -= RHS[i - 1] * Aim[i] / Aii[i - 1];
#endif
  }
  //TUAN DEBUG TUAN 
#if defined(DEBUG_COMPARTMENT) || defined(DEBUG_ASSERT) 
  for (int i = 0; i < size; i++)  // for each compartment on that branch
  {
    if ((Aii[i] != Aii[i]) or (std::fabs(Aii[i]) < SMALL))
    {
      printDebugHH();
      printDebugHHCurrent(i);
      assert(0);
    }
    assert(Aii[i] == Aii[i]);
    if (RHS[i] != RHS[i])
    {
      printDebugHH();
      printDebugHHCurrent(i);
    }
    assert(RHS[i] == RHS[i]);
  }
#endif  //END DEBUG SECTION
}  // end doForwardSolve

// INPUT:
//    Vcur[] at time t
//    all entries of upper-diagonal matrix at time t+dt/2
//      M * V = R.H.S.
//
// Update: Vnew[] at time (t+dt/2)
// Thomas algorithm backward step  (backward Euler for time-step dt/2)
//   - backward substitution on upper triangular matrix 
//   - traverse in the opposite direction from that in :doForwardSolve()
//     i.e. from proximal (index=numCpts-1) down to distal (index=0)
void HodgkinHuxleyVoltage::doBackwardSolve()
{
  unsigned numCpts = branchData->size;
  //TUAN DEBUG
#ifdef DEBUG_COMPARTMENT
  volatile unsigned nidx = _segmentDescriptor.getNeuronIndex(branchData->key);
  volatile unsigned bidx = _segmentDescriptor.getBranchIndex(branchData->key);
  volatile unsigned iteration = getSimulation().getIteration();
#endif
  //END TUAN DEBUG
  if (isProximalCase0)
  {//no proximal-side adjacent cpt, i.e. no Aip
    Vnew[numCpts - 1] = RHS[numCpts - 1] / Aii[numCpts - 1];
  }
  else
  {
#ifdef USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION
    assert(0);
    dyn_var_t V0;
    if (isProximalCase1)
    {
      dyn_var_t w1 = 1.0/(proximalDimension->length);
      dyn_var_t w2 = 1.0/(dimensions[numCpts-1]->length);
      V0 = (w1 * *proximalVoltage + w2 * Vnew[numCpts-1])/(w1+w2);
    }else
    {
     V0 = *proximalVoltage; 
    }
    Vnew[numCpts - 1] =
        (RHS[numCpts - 1] - Aip[numCpts - 1] * V0) / Aii[numCpts - 1];
#else
    Vnew[numCpts - 1] =
        (RHS[numCpts - 1] - Aip[numCpts - 1] * *proximalVoltage) / Aii[numCpts - 1];
#endif
  }
  //IMPORTANT: solve the direction  opposite to that in :doForwardSolve()
  for (int i = numCpts - 2; i >= 0; i--)
  {
    Vnew[i] = (RHS[i] - Aip[i] * Vnew[i + 1]) / Aii[i];
  }
#if defined(DEBUG_HH) || defined(DEBUG_ASSERT)
  for (int i = 0; i < numCpts; ++i)
  {
    if (not (Vnew[i] == Vnew[i])
        //			or std::fabs(Vnew[i]-Vold)/(*getSharedMembers().deltaT)
        or Vnew[i]> 230.0
        or Vnew[i] < -330.0
       )
    {
      printDebugHH(i);
#ifdef DEBUG_ASSERT
      assert(0);
      assert(Vnew[i] == Vnew[i]);  // making sure Vnew[i] is not NaN
#endif
    }
  }
#endif
}  // end doBackwardSolve

//NOTE: a is the current compartment, and
//      b is the parent compartment (proximal side)
// unit: [nS/um^2]
dyn_var_t HodgkinHuxleyVoltage::getLambda_parent(DimensionStruct* a,
    DimensionStruct* b,
    int index,  // of 'a'
    bool connectJunction /* if 'b' is junction*/)
{
  dyn_var_t Rb ;// radius_middle ()
//#ifdef NEW_DISTANCE_NONUNIFORM_GRID 
//  dyn_var_t dsi = getHalfDistance(index);
//#else
//  dyn_var_t dsi = a->length;
//#endif
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
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
    if (connectJunction) Rb = b->r;
    else Rb = 0.5 * (a->r + b->r);
#else
    Rb = 0.5 * (a->r + b->r);
#endif
    distance = std::fabs(b->dist2soma - a->dist2soma);
  }
  dyn_var_t Gaxial = M_PI * Rb * Rb/(getSharedMembers().Ra * distance ); // [nS]
  return (Gaxial / a->surface_area); //IMPORTANT: this is the right one
  //return (Rb * Rb /
  //    (2.0 * getSharedMembers().Ra * dsi * distance * a->r)); /* needs fixing */
  //GOAL:
  // return da / (4 * Ra * (lengthA+lengthB)/2 * lengthA) --> if b is parent (proximal)
  // return db / (4 * Ra * (lengthA+lengthB)/2 * lengthB) --> if b is child
  // da = diameter of 'a'; 'db' = diameter of 'b'
  //return (2*Rb) / (4 * getSharedMembers().Ra * distance *  a->length);
}
//NOTE: a is the current compartment, and
//      b is the children compartment (distal side)
// unit: [nS/um^2]
dyn_var_t HodgkinHuxleyVoltage::getLambda_child(DimensionStruct* a,
    DimensionStruct* b,
    int index,  // of 'a'
    bool connectJunction /* if 'b' is junction*/)
{
  dyn_var_t Rb ;// radius_middle ()
//#ifdef NEW_DISTANCE_NONUNIFORM_GRID 
//  dyn_var_t dsi = getHalfDistance(index);
//#else
//  dyn_var_t dsi = a->length;
//#endif
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
  {//a  CAN't BE the compartment representing 'soma'
    assert(0);
  }
  else if (b->dist2soma <= SMALL)
  {//b  CAN't BE the compartment representing 'soma'
    assert(0);
  }
  else
  {
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
    if (connectJunction) Rb = b->r;
    else Rb = 0.5 * (a->r + b->r);
#else
    Rb = 0.5 * (a->r + b->r);
#endif
    distance = std::fabs(b->dist2soma - a->dist2soma);
  }
  dyn_var_t Gaxial = M_PI * Rb * Rb/(getSharedMembers().Ra * distance ); // [nS]
  return (Gaxial / a->surface_area);
  //return (Rb * Rb /
  //    (2.0 * getSharedMembers().Ra * dsi * distance * a->r)); /* needs fixing */
  //GOAL:
  // return da / (4 * Ra * (lengthA+lengthB)/2 * lengthA) --> if b is parent (proximal)
  // return db / (4 * Ra * (lengthA+lengthB)/2 * lengthB) --> if b is child
  // da = diameter of 'a'; 'db' = diameter of 'b'
  //return (2*Rb) / (4 * getSharedMembers().Ra * distance *  b->length);
}
//NOTE: a is the current compartment, and
//      b is the adjacent compartment (can be proximal or distal side)
// unit: [nS/um^2]
dyn_var_t HodgkinHuxleyVoltage::getLambda(DimensionStruct* a,
    DimensionStruct* b,
    int index,  // of 'a'
    bool connectJunction /* if 'b' is junction*/)
{
  dyn_var_t Rb ;// radius_middle ()
//#ifdef NEW_DISTANCE_NONUNIFORM_GRID 
//  dyn_var_t dsi = getHalfDistance(index);
//#else
//  dyn_var_t dsi = a->length;
//#endif
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
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
    if (connectJunction) Rb = b->r;
    else Rb = 0.5 * (a->r + b->r);
#else
    Rb = 0.5 * (a->r + b->r);
#endif
    distance = std::fabs(b->dist2soma - a->dist2soma);
  }
  dyn_var_t Gaxial = M_PI * Rb * Rb/(getSharedMembers().Ra * distance ); // [nS]
  return (Gaxial / a->surface_area);
  //return (Rb * Rb /
  //    (2.0 * getSharedMembers().Ra * dsi * distance * a->r)); /* needs fixing */
}

//find the lambda between the terminal point of the 
//compartment represented by 'a'
//'a' can be cpt[0] (distal-end) or cpt[size-1] (proximal-end)
dyn_var_t HodgkinHuxleyVoltage::getLambda(DimensionStruct* a, int index)
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
  return (Rb * Rb /
      (2.0 * getSharedMembers().Ra * dsi * distance * a->r)); /* needs fixing */
}

// GOAL: Get coefficient of Aip[0] and Aim[size-1] of unit [nS/um^2]
//  for Vm(i=0,j=branch-index)
// i.e. at implicit branch point, of the current compartment 'i'=0
//          for every distal-branch 'j'
//  Aij = 1/A * (pi*r_branch^2/(Ra * ds_branch))
// given
//  A = surface_area
//  NOTE: 'a' is the distal-end compartment of the branch (i=0)
//        serving as implicit branch 
dyn_var_t HodgkinHuxleyVoltage::getAij(DimensionStruct* a, DimensionStruct* b,
                                       dyn_var_t A,
                                       bool connectJunction)
{
  dyn_var_t Rb;
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
  {
    assert(0); // a CANNOT be soma
  }
  else if (b->dist2soma <= SMALL)
  {//b is soma
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA;
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
    distance = fabs(b->dist2soma - a->dist2soma);
  }
  return (M_PI * Rb * Rb /
      (A * getSharedMembers().Ra * distance));
}
// When the current node/compartment is an implicit junction
// Then conservational rule is applied for that node
// // GOAL: Get coefficient of Aip[size-1] of unit [nS/um^2]
// i.e. at implicit branch point, of the current compartment 'i'=0
//          for every distal-branch 'j'
//  Aij = 1/A * (pi*r_branch^2/(Ra * ds_branch))
// given
//  A = surface_area of node 'a'
//  r_branhch = radisus of interface between nodes 'a' and 'b'
//  ds_branch = distance between 2 nodes 
//  NOTE: 'a' is the distal-end compartment of the branch (i=0)
//        serving as implicit branch 
dyn_var_t HodgkinHuxleyVoltage::getAij_parent(DimensionStruct* a, DimensionStruct* b,
                                       dyn_var_t A,
                                       bool connectJunction)
{
  dyn_var_t Rb;
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
  {
    assert(0); // a CANNOT be soma
  }
  else if (b->dist2soma <= SMALL)
  {//b is soma
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA;
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
  return (M_PI * Rb * Rb /
      (A * getSharedMembers().Ra * distance));
}
// When the current node/compartment is an implicit junction
// Then conservational rule is applied for that node
// // GOAL: Get coefficient of Aim[size-1] of unit [nS/um^2]
// i.e. at implicit branch point, of the current compartment 'i'=0
//          for every distal-branch 'j'
//  Aij = 1/A * (pi*r_branch^2/(Ra * ds_branch))
// given
//  A = surface_area of node 'a'
//  r_branhch = radisus of interface between nodes 'a' and 'b'
//  ds_branch = distance between 2 nodes 
//  NOTE: 'a' is the distal-end compartment of the branch (i=0)
//        serving as implicit branch 
dyn_var_t HodgkinHuxleyVoltage::getAij_child(DimensionStruct* a, DimensionStruct* b,
                                       dyn_var_t A,
                                       bool connectJunction)
{
  dyn_var_t Rb;
  dyn_var_t distance;
  if (a->dist2soma <= SMALL)
  {
    assert(0); // a CANNOT be soma
  }
  else if (b->dist2soma <= SMALL)
  {//b is soma
    Rb = a->r;
#ifdef USE_SCALING_NECK_FROM_SOMA
    //TEST 
    Rb /= SCALING_NECK_FROM_SOMA;
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
  return (M_PI * Rb * Rb /
      (A * getSharedMembers().Ra * distance));
}


void HodgkinHuxleyVoltage::setReceptorCurrent(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset,
    CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(receptorCurrents.size() > 0);
#endif
  receptorCurrents[receptorCurrents.size() - 1].index = CG_inAttrPset->idx;
}

// to be called at connection-setup time
//    check MDL for what kind of connection then it is called
void HodgkinHuxleyVoltage::setInjectedCurrent(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset,
    CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(injectedCurrents.size() > 0);
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
        CurrentProducer* CG_CurrentProducerPtr =
            dynamic_cast<CurrentProducer*>(CG_variable);
        if (CG_CurrentProducerPtr == 0)
        {
          std::cerr << "Dynamic Cast of CurrentProducer failed in "
                       "HodgkinHuxleyVoltage" << std::endl;
          exit(-1);
        }
        injectedCurrents.increase();
        injectedCurrents[injectedCurrents.size() - 1].current =
            CG_CurrentProducerPtr->CG_get_CurrentProducer_current();
        injectedCurrents[injectedCurrents.size() - 1].index = i;
        checkAndAddPreVariable(CG_variable);
      }
    }
  }
  else if (CG_inAttrPset->idx < 0)  // if we pass in the InAttrPset with 'idx' attribute 
  {//with a negative value, i.e. [passed via 'Probe' of TissueFunctor] 
     // then inject at all compartments in that ComputeBranch (CB)
    injectedCurrents[injectedCurrents.size() - 1].index = 0;
    for (int i = 1; i < branchData->size; ++i)
    {
      CurrentProducer* CG_CurrentProducerPtr =
          dynamic_cast<CurrentProducer*>(CG_variable);
      if (CG_CurrentProducerPtr == 0)
      {
        std::cerr
          << "Dynamic Cast of CurrentProducer failed in HodgkinHuxleyVoltage"
          << std::endl;
        exit(-1);
      }
      injectedCurrents.increase();
      injectedCurrents[injectedCurrents.size() - 1].current =
          CG_CurrentProducerPtr->CG_get_CurrentProducer_current();
      injectedCurrents[injectedCurrents.size() - 1].index = i;
      checkAndAddPreVariable(CG_variable);
    }
  }
  else
  {//i.e. bi-directional connection (electrical synapse or spineneck-compartment)
   //NOTE: The current component already been assigned via code-generated specified in MDL
    injectedCurrents[injectedCurrents.size() - 1].index = CG_inAttrPset->idx;
  }
}

HodgkinHuxleyVoltage::~HodgkinHuxleyVoltage() {}

dyn_var_t HodgkinHuxleyVoltage::getHalfDistance (int index) 
{
  dyn_var_t halfDist = 0.0 ;
  unsigned size = branchData->size;  //# of compartments
  assert(index >=0 and index <= size-1);
  if  (index == size-1)
  {
    if (! isProximalCase0)
    {
      if (proximalDimension->dist2soma <= SMALL)
      {//proximal-cpt is the SOMA
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
  {
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

#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION1
void HodgkinHuxleyVoltage::updateSpineCount(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset) 
{
  unsigned size = branchData->size;  //# of compartments
  if (countSpineConnected.size() != size) 
  {
    countSpineConnected.increaseSizeTo(size);
    for (int i = 0; i < size; i++)
      countSpineConnected[i] = 0;
  }
  countSpineConnected[CG_inAttrPset->idx]++;
}

void HodgkinHuxleyVoltage::updateGapJunctionCount(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_HodgkinHuxleyVoltageInAttrPSet* CG_inAttrPset, CG_HodgkinHuxleyVoltageOutAttrPSet* CG_outAttrPset) 
{
  unsigned size = branchData->size;  //# of compartments
  if (countGapJunctionConnected.size() != size) 
  {
    countGapJunctionConnected.increaseSizeTo(size); 
    for (int i = 0; i < size; i++)
      countGapJunctionConnected[i] = 0;
  }
  countGapJunctionConnected[CG_inAttrPset->idx]++;
}
#endif

