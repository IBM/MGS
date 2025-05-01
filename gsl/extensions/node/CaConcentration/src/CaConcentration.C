// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "CaConcentration.h"
#include "CG_CaConcentration.h"
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

SegmentDescriptor CaConcentration::_segmentDescriptor;

// NOTE: value = 1e6/(zCa*Farad)
// zCa = valence of Ca2+
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

#if CALCIUM_CYTO_DYNAMICS == FAST_BUFFERING
#define DCa (getSharedMembers().DCaeff)
#else
#define DCa (getSharedMembers().DCa)
#endif

//#define DEBUG_HH
// Conserved region (only change ClassName)
//{{{
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
//void CaConcentration::forwardSolve0(RNG& rng)
//{
//  if (computeOrder == 0)
//  {
//    doForwardSolve();
//  }
//}
//void CaConcentration::backwardSolve0(RNG& rng)
//{
//  if (computeOrder == 0)
//  {
//    doBackwardSolve();
//  }
//}
void CaConcentration::backwardSolve0_corrector(RNG& rng)
{
  if (computeOrder == 0)
  {
    doBackwardSolve();
  }
}
void CaConcentration::forwardSolve0_corrector(RNG& rng)
{
  if (computeOrder == 0)
  {
    doForwardSolve_corrector();
  }
}
// Recalculate: RHS[], Aii[]  at time (t +dt/2)
// Unit: RHS =  [uM/msec]
//       Aii =  [1/msec]
// Thomas algorithm forward step 
// IMPORTANT: Ca_new[] has been changed from previous 'doFowardSolve, doBackwardSolve'
//   so we need to use Ca_cur[] in this function
void CaConcentration::doForwardSolve_corrector()
{
  unsigned numCpts = branchData->size;

  //Find A[ii]i and RHS[ii]  
  //  1. ionic currents 
  for (int i = 0; i < numCpts; i++)
  {
    RHS[i] = getSharedMembers().bmt * Ca_cur[i];
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
    // loop through different kinds of Ca2+ currents (LCCv12, LCCv13, R-type, ...)
    // 1.a. producing I_Ca [pA/um^2]
    Array<ChannelCaCurrents>::iterator iter = channelCaCurrents.begin();
    Array<ChannelCaCurrents>::iterator end = channelCaCurrents.end();
    for (; iter != end; iter++)
    {
      RHS[i] -= currentDensityToConc[i] * (*iter->currents)[i];
    }

    // 1.b. producing J_Ca [uM/msec]
    Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
    Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
    for (; fiter != fend; fiter++)
    {
      RHS[i] +=  (*fiter->fluxes)[i];
    }
    /* This is a simple implementation of calcium extrusion. To be elaborated as
     * needed. */
    // TUAN: need to be updated to take into account PMCA
    //RHS[i] -= CaClearance * (Ca_cur[i] - getSharedMembers().CaBaseline);
  }

  //  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<ReceptorCaCurrent>::iterator riter = receptorCaCurrents.begin();
  Array<ReceptorCaCurrent>::iterator rend = receptorCaCurrents.end();
  for (; riter != rend; riter++)
  {
    int i = riter->index;
    RHS[i] -= currentDensityToConc[i] * *(riter->current);
  }

    // 1.c. HH-like of concentration diffusion
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  Array<TargetAttachCaConcentration >::iterator ciiter = targetAttachCaConcentration.begin();
  Array<TargetAttachCaConcentration >::iterator ciend = targetAttachCaConcentration.end();
  //dyn_var_t invTime = 1.0/(getSharedMembers().dt;
  for (; ciiter != ciend; ciiter++)
  {
    int i = (ciiter)->index;
    RHS[i] += (*(ciiter->inverseTime)) * (*(ciiter->Ca)); //[uM/ms]
    Aii[i] += (*(ciiter->inverseTime)) ; //[1/ms]
  }
#endif
  
  //  4. injected currents (pA)
  Array<InjectedCaCurrent>::iterator iiter = injectedCaCurrents.begin();
  Array<InjectedCaCurrent>::iterator iend = injectedCaCurrents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < numCpts)
      RHS[iiter->index] += *(iiter->current)  * iiter->currentToConc;
  }

  /* * *  Forward Solve Ax = B * * */
  /* Starting from distal-end (i=0)
   * Eliminate Aim[?] by taking
   * RHS -= Aim[?] * V[proximal]
   * Aii = 
   */
  for (int i = 1; i < numCpts; i++)
  {
    Aii[i] -= Aip[i - 1] * Aim[i] / Aii[i - 1];
    RHS[i] -= RHS[i - 1] * Aim[i] / Aii[i - 1];
  }

#ifdef MICRODOMAIN_CALCIUM
  //must put here
  if (microdomainNames.size() > 0)
    updateMicrodomains();
#endif
}
#endif
void CaConcentration::solve(RNG& rng)
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
void CaConcentration::forwardSolve1(RNG& rng)
{
  if (computeOrder == 1)
  {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve1(RNG& rng)
{
  if (computeOrder == 1) doBackwardSolve();
}
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
void CaConcentration::backwardSolve1_corrector(RNG& rng)
{
  if (computeOrder == 1)
  {
    doBackwardSolve();
  }
}
void CaConcentration::forwardSolve1_corrector(RNG& rng)
{
  if (computeOrder == 1)
  {
    doForwardSolve_corrector();
  }
}
#endif
#endif

#if MAX_COMPUTE_ORDER > 1
void CaConcentration::forwardSolve2(RNG& rng)
{
  if (computeOrder == 2)
  {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve2(RNG& rng)
{
  if (computeOrder == 2) doBackwardSolve();
}
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
void CaConcentration::backwardSolve2_corrector(RNG& rng)
{
  if (computeOrder == 2)
  {
    doBackwardSolve();
  }
}
void CaConcentration::forwardSolve2_corrector(RNG& rng)
{
  if (computeOrder == 2)
  {
    doForwardSolve_corrector();
  }
}
#endif
#endif

#if MAX_COMPUTE_ORDER > 2
void CaConcentration::forwardSolve3(RNG& rng)
{
  if (computeOrder == 3)
  {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve3(RNG& rng)
{
  if (computeOrder == 3) doBackwardSolve();
}
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
void CaConcentration::backwardSolve3_corrector(RNG& rng)
{
  if (computeOrder == 3)
  {
    doBackwardSolve();
  }
}
void CaConcentration::forwardSolve3_corrector(RNG& rng)
{
  if (computeOrder == 3)
  {
    doForwardSolve_corrector();
  }
}
#endif
#endif

#if MAX_COMPUTE_ORDER > 3
void CaConcentration::forwardSolve4(RNG& rng)
{
  if (computeOrder == 4)
  {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve4(RNG& rng)
{
  if (computeOrder == 4) doBackwardSolve();
}
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
void CaConcentration::backwardSolve4_corrector(RNG& rng)
{
  if (computeOrder == 4)
  {
    doBackwardSolve();
  }
}
void CaConcentration::forwardSolve4_corrector(RNG& rng)
{
  if (computeOrder == 4)
  {
    doForwardSolve_corrector();
  }
}
#endif
#endif

#if MAX_COMPUTE_ORDER > 4
void CaConcentration::forwardSolve5(RNG& rng)
{
  if (computeOrder == 5)
  {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve5(RNG& rng)
{
  if (computeOrder == 5) doBackwardSolve();
}
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
void CaConcentration::backwardSolve5_corrector(RNG& rng)
{
  if (computeOrder == 5)
  {
    doBackwardSolve();
  }
}
void CaConcentration::forwardSolve5_corrector(RNG& rng)
{
  if (computeOrder == 5)
  {
    doForwardSolve_corrector();
  }
}
#endif
#endif

#if MAX_COMPUTE_ORDER > 5
void CaConcentration::forwardSolve6(RNG& rng)
{
  if (computeOrder == 6)
  {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve6(RNG& rng)
{
  if (computeOrder == 6) doBackwardSolve();
}
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
void CaConcentration::backwardSolve6_corrector(RNG& rng)
{
  if (computeOrder == 6)
  {
    doBackwardSolve();
  }
}
void CaConcentration::forwardSolve6_corrector(RNG& rng)
{
  if (computeOrder == 6)
  {
    doForwardSolve_corrector();
  }
}
#endif
#endif

#if MAX_COMPUTE_ORDER > 6
void CaConcentration::forwardSolve7(RNG& rng)
{
  if (computeOrder == 7)
  {
    doForwardSolve();
  }
}

void CaConcentration::backwardSolve7(RNG& rng)
{
  if (computeOrder == 7) doBackwardSolve();
}
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR
void CaConcentration::backwardSolve7_corrector(RNG& rng)
{
  if (computeOrder == 7)
  {
    doBackwardSolve();
  }
}
void CaConcentration::forwardSolve7_corrector(RNG& rng)
{
  if (computeOrder == 7)
  {
    doForwardSolve_corrector();
  }
}
#endif
#endif

bool CaConcentration::confirmUniqueDeltaT(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
    CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}

//TUAN: TODO challenge
//   how to check for 2 sites overlapping
//   if we don't retain the dimension's (x,y,z) coordinate
//  Even if we retain (x,y,z) this value change with the #capsule per compartment
//   and geometric sampling --> so not a good choice
bool CaConcentration::checkSite(const CustomString& CG_direction,
                                const CustomString& CG_component,
                                NodeDescriptor* CG_node, Edge* CG_edge,
                                VariableDescriptor* CG_variable,
                                Constant* CG_constant,
                                CG_CaConcentrationInAttrPSet* CG_inAttrPset,
                                CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
  TissueSite& site = CG_inAttrPset->site;
  bool atSite = (site.r == 0);
  for (unsigned int i = 0; !atSite && i < dimensions.size(); ++i)
    atSite = ((site.r * site.r) >= DISTANCE_SQUARED(&site, dimensions[i]));
  return atSite;
}

void CaConcentration::setProximalJunction(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
    CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
  proximalJunction = true;
}

// update: Ca(t+dt) = 2 * Ca(t+dt/2) - Ca(t)
// second-step (final step) in Crank-Nicolson method
void CaConcentration::finish(RNG& rng)
{
  unsigned size = branchData->size;
#ifdef DEBUG_HH
  printDebugHH();
#endif
  for (int i = 0; i < size; ++i)
  {
    Ca_cur[i] = Ca_new[i] = 2.0 * Ca_new[i] - Ca_cur[i];
#ifdef DEBUG_ASSERT
    if (Ca_new[i] != Ca_new[i] or 
        Ca_new[i] <= 0)
      printDebugHH();
    assert(Ca_new[i] >= 0);
    assert(Ca_new[i] == Ca_new[i]);  // making sure Ca_new[i] is not NaN
#endif
  }
#ifdef MICRODOMAIN_CALCIUM
  //must put here
  int numCpts = branchData->size;
  for (unsigned int ii = 0; ii < microdomainNames.size(); ii++)
  {//calculate RHS[] and Ca_microdomain[]
    int offset = ii * numCpts;
    for (int jj = 0; jj < numCpts; jj++ )
    {
      Ca_microdomain_cur[jj+offset] = Ca_microdomain[jj+offset] = 
        2 * Ca_microdomain[jj+offset]  - Ca_microdomain_cur[jj+offset];
    }
  }
#endif
}

// Get cytoplasmic surface area (um^2)
// at the compartment based on its index 'i'
dyn_var_t CaConcentration::getArea(int i) // Tuan: check ok
{
  dyn_var_t area= 0.0;
  area = dimensions[i]->surface_area * FRACTION_SURFACEAREA_CYTO;
  return area;
}

// Get cytoplasmic volume (um^3) 
// at the compartment based on its index 'i'
dyn_var_t CaConcentration::getVolume(int i) // Tuan: check ok
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
void CaConcentration::initializeCompartmentData(RNG& rng)
{
  // for a given computing process:
  //  here all the data in vector-form are initialized to
  //  the same size as the number of compartments in a branch (i.e. branchData)
  unsigned numCpts = branchData->size;  //# of compartments
  computeOrder = _segmentDescriptor.getComputeOrder(branchData->key);

  if (isProximalCase2) assert(computeOrder == 0);
  if (isDistalCase2) assert(computeOrder == MAX_COMPUTE_ORDER);
  assert(dimensions.size() == numCpts);
  assert(Ca_new.size() == numCpts);
  assert(distalDimensions.size() == distalInputs.size());

  // allocate data
  if (Ca_cur.size() != numCpts) Ca_cur.increaseSizeTo(numCpts);
  if (Aii.size() != numCpts) Aii.increaseSizeTo(numCpts);
  if (Aip.size() != numCpts) Aip.increaseSizeTo(numCpts);
  if (Aim.size() != numCpts) Aim.increaseSizeTo(numCpts);
  if (RHS.size() != numCpts) RHS.increaseSizeTo(numCpts);
  if (currentDensityToConc.size() != numCpts) currentDensityToConc.increaseSizeTo(numCpts);
#ifdef MICRODOMAIN_CALCIUM
  //assert(0); //add data here Ca_microdomain
  //NOTE: already allocated in createMicrodomainData()
#endif

  // initialize data
  Ca_cur[0] = Ca_new[0];
  for (int i = 1; i < numCpts; ++i)
  {
    Ca_new[i] = Ca_new[0];
    Ca_cur[i] = Ca_cur[0];
  }
#ifdef MICRODOMAIN_CALCIUM
  unsigned int ii=0;
  for (ii = 0; ii < Ca_microdomain.size(); ii++)
  {
    Ca_microdomain[ii] = Ca_cur[0];
    Ca_microdomain_cur[ii] = Ca_cur[0];
  }
#endif
  // go through each compartments in a branch
  for (int i = 0; i < numCpts; ++i)
  {
    Aii[i] = Aip[i] = Aim[i] = RHS[i] = 0.0;
    currentDensityToConc[i] = getArea(i) * uM_um_cubed_per_pA_msec / getVolume(i);
  }
#ifdef MICRODOMAIN_CALCIUM
  //for (unsigned int ii=0; ii < microdomainNames.size(); ++ii)
  //{
  //  int offset = ii * numCpts;
  //  for (int jj = 0; jj < numCpts; jj++ )
  //  {
  //    currentDensityToConc_microdomain[offset+jj] = getArea(jj) * uM_um_cubed_per_pA_msec / 
  //      volume_microdomain[offset+jj];
  //  }
  //}
#endif

  // go through different kinds of injected Calcium currents
  //   one of which is the bidirectional current from spine neck
  Array<InjectedCaCurrent>::iterator iiter = injectedCaCurrents.begin();
  Array<InjectedCaCurrent>::iterator iend = injectedCaCurrents.end();
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
      //Aip[0] = -getAij(dimensions[0], proximalDimension, volume, true);
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
      //Aij.push_back(-getAij(dimensions[0], distalDimensions[n], volume, true));
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

void CaConcentration::printDebugHH()
{
  unsigned size = branchData->size;
  for (int i = 0; i < size; ++i)
  {
    this->printDebugHH(i);
  }
}

void CaConcentration::printDebugHH(int cptIndex)
{
  unsigned size = branchData->size;
  if (cptIndex == 0)
  {
    std::cerr << "iter,time| BRANCH [rank, nodeIdx, layerIdx, cptIdx]"
      << "(neuronIdx, brIdx, brOrder, brType) distal(C0 | C1 | C2 | C3) :"
      << " prox( C0 | C1 | C2) |"
      << "{x,y,z,r, dist2soma, surface_area, volume, length} Cai\n";
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
    << Ca_new[i]  << " " << std::endl;
}


// Update: RHS[], Aii[]  at time (t +dt/2)
// Unit: RHS =  [uM/msec]
//       Aii =  [1/msec]
// Thomas algorithm forward step 
void CaConcentration::doForwardSolve()
{
  unsigned numCpts = branchData->size;

  //Find A[ii]i and RHS[ii]  
  //  1. ionic currents 
  for (int i = 0; i < numCpts; i++)
  {
    RHS[i] = getSharedMembers().bmt * Ca_cur[i];
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
    // loop through different kinds of Ca2+ currents (LCCv12, LCCv13, R-type, ...)
    // 1.a. producing I_Ca [pA/um^2]
    Array<ChannelCaCurrents>::iterator iter = channelCaCurrents.begin();
    Array<ChannelCaCurrents>::iterator end = channelCaCurrents.end();
    for (; iter != end; iter++)
    {
      RHS[i] -= currentDensityToConc[i] * (*iter->currents)[i];
    }

    // 1.b. producing J_Ca [uM/msec]
    Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
    Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
    for (; fiter != fend; fiter++)
    {
      RHS[i] +=  (*fiter->fluxes)[i];
    }
    /* This is a simple implementation of calcium extrusion. To be elaborated as
     * needed. */
    // TUAN: need to be updated to take into account PMCA
    //RHS[i] -= CaClearance * (Ca_cur[i] - getSharedMembers().CaBaseline);
  }

  //  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<ReceptorCaCurrent>::iterator riter = receptorCaCurrents.begin();
  Array<ReceptorCaCurrent>::iterator rend = receptorCaCurrents.end();
  for (; riter != rend; riter++)
  {
    int i = riter->index;
    RHS[i] -= currentDensityToConc[i] * *(riter->current);
  }

    // 1.c. HH-like of concentration diffusion
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  Array<TargetAttachCaConcentration >::iterator ciiter = targetAttachCaConcentration.begin();
  Array<TargetAttachCaConcentration >::iterator ciend = targetAttachCaConcentration.end();
  //dyn_var_t invTime = 1.0/(getSharedMembers().dt;
  for (; ciiter != ciend; ciiter++)
  {
    int i = (ciiter)->index;
    RHS[i] += (*(ciiter->inverseTime)) * (*(ciiter->Ca)); //[uM/ms]
    Aii[i] += (*(ciiter->inverseTime)) ; //[1/ms]
  }
#endif
  
  //  4. injected currents (pA)
  Array<InjectedCaCurrent>::iterator iiter = injectedCaCurrents.begin();
  Array<InjectedCaCurrent>::iterator iend = injectedCaCurrents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < numCpts)
      RHS[iiter->index] += *(iiter->current)  * iiter->currentToConc;
  }

  /* * *  Forward Solve Ax = B * * */
  /* Starting from distal-end (i=0)
   * Eliminate Aim[?] by taking
   * RHS -= Aim[?] * V[proximal]
   * Aii = 
   */
  for (int i = 1; i < numCpts; i++)
  {
    Aii[i] -= Aip[i - 1] * Aim[i] / Aii[i - 1];
    RHS[i] -= RHS[i - 1] * Aim[i] / Aii[i - 1];
  }

#ifdef MICRODOMAIN_CALCIUM
  //must put here
  if (microdomainNames.size() > 0)
    updateMicrodomains();
#endif
}

// Update; Ca_new[] at time (t + dt/2)
// Thomas algorithm backward step 
//   - backward substitution on upper triangular matrix
// Next it calls :finish()
void CaConcentration::doBackwardSolve()
{
  unsigned size = branchData->size;
  if (isProximalCase0)
  {
    Ca_new[size - 1] = RHS[size - 1] / Aii[size - 1];
  }
  else
  {
    Ca_new[size - 1] =
        (RHS[size - 1] - Aip[size - 1] * *proximalCaConcentration) /
        Aii[size - 1];
  }
  for (int i = size - 2; i >= 0; i--)
  {
    Ca_new[i] = (RHS[i] - Aip[i] * Ca_new[i + 1]) / Aii[i];
  }
#ifdef MICRODOMAIN_CALCIUM
  //must put here
  if (microdomainNames.size() > 0)
    updateMicrodomains_Ca();
#endif
}

//GOAL: get coefficient of Aip or Aim
//  DCa * (r_{i->j})^2 / (dist^2 * b->r^2)
//NOTE: a is the current compartment, and
//      b is the parent compartment (proximal side)
//      index = index of 'a'
dyn_var_t CaConcentration::getLambda_parent(DimensionStruct* a, 
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
  return (M_PI * Rb * Rb * DCa) / 
      (volume * distance);
}
//GOAL: get coefficient of Aip or Aim
//  DCa * (r_{i->j})^2 / (dist^2 * b->r^2)
//NOTE: a is the current compartment, and
//      b is the child compartment (distal side)
//      index = index of 'a'
dyn_var_t CaConcentration::getLambda_child(DimensionStruct* a, 
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
  return (M_PI * Rb * Rb * DCa) / 
      (volume * distance);
}
//GOAL: get coefficient of Aip or Aim
//  DCa * (r_{i->j})^2 / (dist^2 * b->r^2)
//NOTE: a is the current compartment, and
//      b is the adjacent compartment (can be proximal or distal side)
dyn_var_t CaConcentration::getLambda(DimensionStruct* a, 
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
    Rb /= SCALING_NECK_FROM_SOMA_WITH;
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
  return (DCa * Rb * Rb /
      (dsi * distance * a->r * a->r)); /* needs fixing */
  /* NOTE: ideally
     return (DCa  /
     (dsi * distance )); 
     */
}
//find the lambda between the terminal point of the 
//compartment represented by 'a'
//'a' can be cpt[0] (distal-end) or cpt[size-1] (proximal-end)
dyn_var_t CaConcentration::getLambda(DimensionStruct* a, int index)
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
  return (DCa * Rb * Rb /
      (dsi * distance * a->r * a->r)); /* needs fixing */
  /* NOTE: ideally
     return (DCa  /
     (dsi * distance )); 
     */
}

// GOAL: Get coefficient of Aip[0] and Aim[size-1]
//  for Cai(i=0,j=branch-index)
// i.e. at implicit branch point
//  DCa * (1/V) * PI * r_(i->j)^2 / (ds_(i->j))
//   V = volume of cytosolic compartment
//   DCa = diffusion constant of Ca(cyto)
//  NOTE: 'a' is the current node;
//        'b' is the proxomal-side node 
dyn_var_t CaConcentration::getAij_parent(DimensionStruct* a, DimensionStruct* b,
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
  return (M_PI * Rb * Rb * DCa /
      (V * distance));
}

// GOAL: Get coefficient of Aip[0] and Aim[size-1]
//  for Cai(i=0,j=branch-index)
// i.e. at implicit branch point
//  DCa * (1/V) * PI * r_(i->j)^2 / (ds_(i->j))
//   V = volume of cytosolic compartment
//   DCa = diffusion constant of Ca(cyto)
//  NOTE: 'a' is the current node;
//        'b' is the distal-side node 
dyn_var_t CaConcentration::getAij_child(DimensionStruct* a, DimensionStruct* b,
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
  return (M_PI * Rb * Rb * DCa /
      (V * distance));
}

// GOAL: Get coefficient of Aip[0] and Aim[size-1]
//  for Cai(i=0,j=branch-index)
// i.e. at implicit branch point
//  DCa * (1/V) * PI * r_(i->j)^2 / (ds_(i->j))
//   V = volume of cytosolic compartment
//   DCa = diffusion constant of Ca(cyto)
//  NOTE: 'a' is the distal-end compartment of the branch (i=0)
//        serving as implicit branch 
dyn_var_t CaConcentration::getAij(DimensionStruct* a, DimensionStruct* b,
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
  return (M_PI * Rb * Rb * DCa /
      (V * distance));
}


void CaConcentration::setReceptorCaCurrent(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
    CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(receptorCaCurrents.size() > 0);
#endif
  receptorCaCurrents[receptorCaCurrents.size() - 1].index = CG_inAttrPset->idx;
}

// to be called at connection-setup time
//    check MDL for what kind of connection then it is called
void CaConcentration::setInjectedCaCurrent(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset,
    CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(injectedCaCurrents.size() > 0);
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
        CaCurrentProducer* CG_CaCurrentProducerPtr =
            dynamic_cast<CaCurrentProducer*>(CG_variable);
        if (CG_CaCurrentProducerPtr == 0)
        {
          std::cerr
              << "Dynamic Cast of CurrentProducer failed in CaConcentration"
              << std::endl;
          exit(-1);
        }
        injectedCaCurrents.increase();
        injectedCaCurrents[injectedCaCurrents.size() - 1].current =
            CG_CaCurrentProducerPtr->CG_get_CaCurrentProducer_current();
        injectedCaCurrents[injectedCaCurrents.size() - 1].index = i;
        checkAndAddPreVariable(CG_variable);
      }
    }
  }
  else if (CG_inAttrPset->idx < 0)  // Can be used via 'Probe' of TissueFunctor
  {//inject at all compartments of one or many branchs meet the condition
    injectedCaCurrents[injectedCaCurrents.size() - 1].index = 0;
    for (int i = 1; i < branchData->size; ++i)
    {
      CaCurrentProducer* CG_CaCurrentProducerPtr =
          dynamic_cast<CaCurrentProducer*>(CG_variable);
      if (CG_CaCurrentProducerPtr == 0)
      {
        std::cerr
          << "Dynamic Cast of CurrentProducer failed in CaConcentration"
          << std::endl;
        exit(-1);
      }
      injectedCaCurrents.increase();
      injectedCaCurrents[injectedCaCurrents.size() - 1].current =
          CG_CaCurrentProducerPtr->CG_get_CaCurrentProducer_current();
      injectedCaCurrents[injectedCaCurrents.size() - 1].index = i;
      checkAndAddPreVariable(CG_variable);
    }
  }
  else
  {//i.e. bi-directional connection (electrical synapse or spineneck-compartment)
   //NOTE: The current component already been assigned via code-generated specified in MDL
    injectedCaCurrents[injectedCaCurrents.size() - 1].index =
        CG_inAttrPset->idx;
  }
}

CaConcentration::~CaConcentration() {}

dyn_var_t CaConcentration::getHalfDistance (int index) 
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
void CaConcentration::setTargetAttachCaConcentration(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(targetAttachCaConcentration.size() > 0);
#endif
  targetAttachCaConcentration[targetAttachCaConcentration.size() - 1].index = CG_inAttrPset->idx;
}
#endif


#ifdef MICRODOMAIN_CALCIUM
void CaConcentration::createMicroDomainData(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset) 
{
  std::string listmicroDomains (CG_inAttrPset->domainName.c_str());
  if (listmicroDomains.empty())
  {//do nothing as no microdomain exists
  }
  else{
    std::vector<std::string> tokens;  // extract all names of microdomains as token
    assert(microdomainNames.size() == 0);
    CustomStringUtils::Tokenize(listmicroDomains, tokens, " ,");
    int numMicrodomains = tokens.size();

    microdomainNames.increaseSizeTo(numMicrodomains);
    v_efflux.increaseSizeTo(numMicrodomains);
#if MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_NTSMACRO
    //checking only
    if (numMicrodomains > 3)
    {
      std::cerr << "ERROR: With _MICRODOMAIN_DATA_FROM_NTSMACRO; we currently support maximum 3 microdomains"
        << std::endl;
      assert(0);
    }
    std::vector<std::string> supportedDomainNames { "domain1", "domain2", "domain3"};
    for (unsigned ii = 0; ii < numMicrodomains; ++ii)
    {
      if (std::find(supportedDomainNames.begin(), supportedDomainNames.end(), tokens[ii]) 
          == supportedDomainNames.end())
      {
        std::cerr << "ERROR: Not-supported domain name: " << tokens[ii] << std::endl;
        std::cerr << "ERROR: With _MICRODOMAIN_DATA_FROM_NTSMACRO, we limit to using these names \n";
        for (auto ii = supportedDomainNames.begin(); ii != supportedDomainNames.end(); ++ii)
        {
          std::cerr << *ii << "\n";
        }
      }
    }
#endif

    int numCpts = branchData->size;
    Ca_microdomain.increaseSizeTo(numMicrodomains * numCpts);
    Ca_microdomain_cur.increaseSizeTo(numMicrodomains * numCpts);
    RHS_microdomain.increaseSizeTo(numMicrodomains * numCpts);
    //currentDensityToConc_microdomain.increaseSizeTo(numMicrodomains * numCpts);
    volume_microdomain.increaseSizeTo(numMicrodomains * numCpts);
    
    for (unsigned ii = 0; ii < numMicrodomains; ++ii)
    {
      CustomString domainName(tokens[ii].c_str());
      microdomainNames[ii] = domainName;
      int offset = ii * numCpts;
#if MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_CHANPARAM
      //domain3  <v_efflux={0.003}; depth_microdomain={10}; fraction_surface={1.0}>
      std::map<std::string, std::vector<float> > 
        domainData = Params::_microdomainArrayParamsMap[tokens[ii]]; 
      if (domainData.count("depth_microdomain") == 0 or 
          domainData.count("fraction_surface") == 0)
      {
        std::cerr << "microdomain " << tokens[ii] << " does not have either depth_microdomain or 'fraction_surface' defined" << std::endl; 
        assert(0); 
      }
      if (domainData.count("v_efflux") == 0)
      {
        std::cerr << "microdomain " << tokens[ii] << " does not have 'v_efflux' defined" << std::endl; 
        assert(0); 
      }
      if (domainData["depth_microdomain"].size() > 1 or 
          domainData["fraction_surface"].size() > 1)
      {
        std::cerr << "microdomain " << tokens[ii] << ": use ONLY 1 value for 'depth_microdomain' and 'fraction_surface' " << std::endl; 
        assert(0); 
      }
      if (domainData["v_efflux"].size() > 1)
      {
        std::cerr << "microdomain " << tokens[ii] << ": use ONLY 1 value for 'v_efflux'" << std::endl; 
        assert(0); 
      }
#endif
      for (int jj = 0; jj < numCpts; jj++ )
      {
#if MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_NTSMACRO
        if (tokens[ii] == "domain1")
        {
          //volume_microdomain[offset+jj] = dimensions[jj]->volume * VOLUME_MICRODOMAIN1;
          volume_microdomain[offset+jj] = dimensions[jj]->surface_area * FRACTION_SURFACEAREA_MICRODOMAIN1 * DEPTH_MICRODOMAIN1 * 1e-3;  // [um^3]
        }
        if (tokens[ii] == "domain2")
        {
          //volume_microdomain[offset+jj] = dimensions[jj]->volume * VOLUME_MICRODOMAIN2;
          volume_microdomain[offset+jj] = dimensions[jj]->surface_area * FRACTION_SURFACEAREA_MICRODOMAIN2 * DEPTH_MICRODOMAIN2 * 1e-3;  // [um^3]
        }
        if (tokens[ii] == "domain3")
        {
          //volume_microdomain[offset+jj] = dimensions[jj]->volume * VOLUME_MICRODOMAIN3;
          volume_microdomain[offset+jj] = dimensions[jj]->surface_area * FRACTION_SURFACEAREA_MICRODOMAIN3 * DEPTH_MICRODOMAIN3 * 1e-3;  // [um^3]
        }
#elif MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_CHANPARAM
        volume_microdomain[offset+jj] = dimensions[jj]->surface_area * domainData["fraction_surface"][0] * domainData["depth_microdomain"][0] * 1e-3;  // [um^3]
#endif
      }
#if MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_NTSMACRO
      //v_efflux[ii] = V_EFFLUX;
      //VOLUME_MICRODOMAIN
      if (tokens[ii] == "domain1")
      {
        v_efflux[ii] = V_EFFLUX_DOMAIN1;
      }
      if (tokens[ii] == "domain2")
      {
        v_efflux[ii] = V_EFFLUX_DOMAIN2;
      }
      if (tokens[ii] == "domain3")
      {
        v_efflux[ii] = V_EFFLUX_DOMAIN3;
      }
#elif MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_CHANPARAM
      v_efflux[ii] = domainData["v_efflux"][0];
#endif
    }
  }
}

void CaConcentration::setupCurrent2Microdomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset) 
{//this current is supposed to project into the Ca-domain with name defined in 'CG_inAttrPset->domainName'
  //put channel producing Ca2+ influx to the right location
  //from that we can update the [Ca2+] in the associated microdomain
  CustomString microdomainName = CG_inAttrPset->domainName;
  int ii = 0;
  while (microdomainNames[ii] != microdomainName)
  {
    ii++;
  }
  _mapCurrentToMicrodomainIndex[channelCaCurrents_microdomain.size()-1] = ii;
}
void CaConcentration::setupFlux2Microdomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationInAttrPSet* CG_inAttrPset, CG_CaConcentrationOutAttrPSet* CG_outAttrPset) 
{//this current is supposed to project into the Ca-domain with name defined in 'CG_inAttrPset->domainName'
  //put channel producing Ca2+ influx to the right location
  //from that we can update the [Ca2+] in the associated microdomain
  CustomString microdomainName = CG_inAttrPset->domainName;
  int ii = 0;
  while (microdomainNames[ii] != microdomainName)
  {
    ii++;
  }
  _mapFluxToMicrodomainIndex[channelCaFluxes_microdomain.size()-1] = ii;
}
//forward
void CaConcentration::updateMicrodomains()
{//Update Aii[] using v_efflux          [1/ms]
  //      RHS[] using v_efflux * Ca_ds  [uM/ms]
  //      and RHS_microdomain[]
  //float LHS = getSharedMembers().bmt; // [1/ms]
  dyn_var_t bmt= getSharedMembers().x_bmt; // [1/ms]
  int numCpts = branchData->size;
  unsigned int ii = 0;
  for (ii = 0; ii < microdomainNames.size(); ii++)
  {
    int offset = ii * numCpts;
    for (int jj = 0; jj < numCpts; jj++)
    {
      RHS_microdomain[jj+offset] = ((bmt * volume_microdomain[ii] / dimensions[jj]->volume) *
         Ca_microdomain[jj+offset]) + v_efflux[ii] * Ca_new[jj];  // [uM/ms]
    }
  }
  Array<ChannelCaCurrents>::iterator citer = channelCaCurrents_microdomain.begin();
  Array<ChannelCaCurrents>::iterator cend = channelCaCurrents_microdomain.end();
  // loop through different kinds of Ca2+ currents (LCCv12, LCCv13, R-type, ...)
  //  I_Ca [pA/um^2]
  ii = 0;
  for (; citer != cend; ++citer, ++ii)
  {
    int offset = _mapCurrentToMicrodomainIndex[ii] * numCpts;
    for (int jj = 0; jj < numCpts; jj++)
    {
      //wrong->RHS_microdomain[offset+jj] -= currentDensityToConc_microdomain[offset+jj] * (*(citer->currents))[jj];  //[uM/ms]
      RHS_microdomain[offset+jj] -= currentDensityToConc[jj] * (*(citer->currents))[jj];  //[uM/ms]
    }
  }
  Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes_microdomain.begin();
  Array<ChannelCaFluxes>::iterator fend = channelCaFluxes_microdomain.end();
  for (; fiter != fend; fiter++)
  {
    int offset = _mapFluxToMicrodomainIndex[ii] * numCpts;
    for (int jj = 0; jj < numCpts; jj++)
    {
      RHS_microdomain[offset+jj] += (*(fiter->fluxes))[jj];  //[uM/ms]
    }
  }

  // ... (continue with similar above code for other type of Ca2+ influx
  //      e.g. )
  for (unsigned int ii = 0; ii < microdomainNames.size(); ii++)
  {//calculate RHS[] and Ca_microdomain[]
    int offset = ii * numCpts;
    for (int jj = 0; jj < numCpts; jj++ )
    {
      //finally [NOTE: calculate Ca_microdomain can move to doBackwardSolve()]
      ////option1 to calculate Ca_microdomain
      //  Ca_microdomain[jj+offset] = (RHS_microdomain[jj+offset] + v_efflux[ii] * Ca_new[jj]) 
      //    / (LHS + v_efflux[ii]);
      ////option2 to calculate Ca_microdomain
      //Ca_microdomain[jj+offset] = (RHS_microdomain[jj+offset] - 
      //    v_efflux[ii]/2.0 * (Ca_microdomain[jj+offset]) + v_efflux[ii] * Ca_new[jj]) 
      //  / (LHS + v_efflux[ii]/2.0);
      //REVISED FORMULA : Tuan - 04/20/2017
      RHS[jj] += v_efflux[ii] * Ca_microdomain[jj+offset];
      Aii[jj] += v_efflux[ii];
      //RHS_microdomain[offset+jj] -= v_efflux[ii] * (Ca_microdomain[jj+offset] - Ca_new[jj]);
    }
  }
}
//backward
void CaConcentration::updateMicrodomains_Ca()
{//Update Ca_microdomain[]
  dyn_var_t bmt = getSharedMembers().x_bmt; // [1/ms] -- use a different buffering for the microdomain
  int numCpts = branchData->size;
  for (unsigned int ii = 0; ii < microdomainNames.size(); ii++)
  {
    int offset = ii * numCpts;
    for (int jj = 0; jj < numCpts; jj++ )
    {
      //finally [NOTE: calculate Ca_microdomain can move to doBackwardSolve()]
      ////option1 to calculate Ca_microdomain
      //  Ca_microdomain[jj+offset] = (RHS_microdomain[jj+offset] + v_efflux[ii] * Ca_new[jj]) 
      //    / (LHS + v_efflux[ii]);
      ////option2 to calculate Ca_microdomain
     //double LHS = bmt * volume_microdomain[ii] / dimensions[jj]->volume; 
     // Ca_microdomain[jj+offset] = (RHS_microdomain[jj+offset] - 
     //     v_efflux[ii]/2.0 * (Ca_microdomain[jj+offset]) + v_efflux[ii] * Ca_new[jj]) 
     //   / (LHS + v_efflux[ii]/2.0);
      double LHS = bmt * volume_microdomain[ii] / dimensions[jj]->volume +
          v_efflux[ii];
     Ca_microdomain[jj+offset] = (RHS_microdomain[jj+offset] / LHS) ;
    }
  }
}

#endif

