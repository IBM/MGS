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
#include "CaERConcentration.h"
#include "CG_CaERConcentration.h"
#include "rndm.h"
#include "GridLayerDescriptor.h"
#include "MaxComputeOrder.h"

#include "Branch.h"
#include "StringUtils.h"
#include <cmath>

SegmentDescriptor CaERConcentration::_segmentDescriptor;

#define DISTANCE_SQUARED(a, b)               \
  ((((a)->x - (b)->x) * ((a)->x - (b)->x)) + \
   (((a)->y - (b)->y) * ((a)->y - (b)->y)) + \
   (((a)->z - (b)->z) * ((a)->z - (b)->z)))

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

#if CALCIUM_ER_DYNAMICS == FAST_BUFFERING
#define DCa (getSharedMembers().DCaeff)
#else
#define DCa (getSharedMembers().DCa)
#endif
//#define DEBUG_HH
// Conserved region (only change ClassName)
//{{{
void CaERConcentration::solve(RNG& rng)
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
void CaERConcentration::forwardSolve1(RNG& rng)
{
  if (computeOrder == 1)
  {
    doForwardSolve();
  }
}

void CaERConcentration::backwardSolve1(RNG& rng)
{
  if (computeOrder == 1) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 1
void CaERConcentration::forwardSolve2(RNG& rng)
{
  if (computeOrder == 2)
  {
    doForwardSolve();
  }
}

void CaERConcentration::backwardSolve2(RNG& rng)
{
  if (computeOrder == 2) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 2
void CaERConcentration::forwardSolve3(RNG& rng)
{
  if (computeOrder == 3)
  {
    doForwardSolve();
  }
}

void CaERConcentration::backwardSolve3(RNG& rng)
{
  if (computeOrder == 3) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 3
void CaERConcentration::forwardSolve4(RNG& rng)
{
  if (computeOrder == 4)
  {
    doForwardSolve();
  }
}

void CaERConcentration::backwardSolve4(RNG& rng)
{
  if (computeOrder == 4) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 4
void CaERConcentration::forwardSolve5(RNG& rng)
{
  if (computeOrder == 5)
  {
    doForwardSolve();
  }
}

void CaERConcentration::backwardSolve5(RNG& rng)
{
  if (computeOrder == 5) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 5
void CaERConcentration::forwardSolve6(RNG& rng)
{
  if (computeOrder == 6)
  {
    doForwardSolve();
  }
}

void CaERConcentration::backwardSolve6(RNG& rng)
{
  if (computeOrder == 6) doBackwardSolve();
}
#endif

#if MAX_COMPUTE_ORDER > 6
void CaERConcentration::forwardSolve7(RNG& rng)
{
  if (computeOrder == 7)
  {
    doForwardSolve();
  }
}

void CaERConcentration::backwardSolve7(RNG& rng)
{
  if (computeOrder == 7) doBackwardSolve();
}
#endif

bool CaERConcentration::confirmUniqueDeltaT(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
    CG_CaERConcentrationOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}

// TUAN: TODO challenge
//   how to check for 2 sites overlapping
//   if we don't retain the dimension's (x,y,z) coordinate
//  Even if we retain (x,y,z) this value change with the #capsule per compartment
//   and geometric sampling --> so not a good choice
bool CaERConcentration::checkSite(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
    CG_CaERConcentrationOutAttrPSet* CG_outAttrPset)
{
  TissueSite& site = CG_inAttrPset->site;
  bool atSite = (site.r == 0);
  for (unsigned int i = 0; !atSite && i < dimensions.size(); ++i)
    atSite = ((site.r * site.r) >= DISTANCE_SQUARED(&site, dimensions[i]));
  return atSite;
}

void CaERConcentration::setProximalJunction(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
    CG_CaERConcentrationOutAttrPSet* CG_outAttrPset)
{
  proximalJunction = true;
}

// update: Ca(t+dt) = 2 * Ca(t+dt/2) - Ca(t)
// second-step (final step) in Crank-Nicolson method
void CaERConcentration::finish(RNG& rng)
{
  unsigned size = branchData->size;
#ifdef DEBUG_HH
  printDebugHH();
#endif
  for (int i = 0; i < size; ++i)
  {
    Ca_cur[i] = Ca_new[i] = 2.0 * Ca_new[i] - Ca_cur[i];
#ifdef DEBUG_ASSERT
	//TUAN DEBUG why CaER drop on spines
    //if ( _segmentDescriptor.getNeuronIndex(branchData->key) == 2)
	//{
	//	printDebugHH();
	//}
	//END TUAN DEBUG
	if ( Ca_cur[i] <= 0 || Ca_cur[i] > 2000)
	{
		printDebugHH();
		StringUtils::wait();
	}

    assert(Ca_new[i] >= 0);
    assert(Ca_new[i] == Ca_new[i]);  // making sure Ca_new[i] is not NaN
#endif
  }
}

// Get cytoplasmic surface area at the compartment based on its index 'i'
dyn_var_t CaERConcentration::getArea(int i)  // Tuan: check ok
{
  dyn_var_t area = 0.0;
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
  {  // soma:
    area = dimensions[i]->surface_area * FRACTION_SURFACEAREA_RoughER;
  }
  else
  {
    area = dimensions[i]->surface_area * FRACTION_SURFACEAREA_SmoothER;
  }
  return area;
}

// Get cytoplasmic volume at the compartment based on its index 'i'
dyn_var_t CaERConcentration::getVolume(int i)  // Tuan: check ok
{
  dyn_var_t volume = 0.0;
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
  {  // soma:
    volume = dimensions[i]->volume * FRACTIONVOLUME_RoughER;
  }
  else
  {
    volume = dimensions[i]->volume * FRACTIONVOLUME_SmoothER;
  }
  return volume;
}
//}}} //end Conserved region

// GOAL: initialize data at each branch
//    the compartments along one branch are indexed from distal (index=0)
//    to the proximal (index=branchData->size-1)
//    so Aim[..] from distal side
//       Aip[..] from proximal side
void CaERConcentration::initializeCompartmentData(RNG& rng)
{
  // for a given computing process:
  //  here all the data in vector-form are initialized to
  //  the same size as the number of compartments in a branch (i.e. branchData)
  unsigned size = branchData->size;  //# of compartments
  SegmentDescriptor segmentDescriptor;
  computeOrder = segmentDescriptor.getComputeOrder(branchData->key);
#ifdef DEBUG_ASSERT
  if (isProximalCase2) assert(computeOrder == 0);
  if (isDistalCase2) assert(computeOrder == MAX_COMPUTE_ORDER);
  assert(dimensions.size() == size);
  assert(Ca_new.size() == size);
  assert(distalDimensions.size() == distalInputs.size());
#endif

  // allocate data
  if (Ca_cur.size() != size) Ca_cur.increaseSizeTo(size);
  if (Aii.size() != size) Aii.increaseSizeTo(size);
  if (Aip.size() != size) Aip.increaseSizeTo(size);
  if (Aim.size() != size) Aim.increaseSizeTo(size);
  if (RHS.size() != size) RHS.increaseSizeTo(size);
  if (currentToConc.size() != size) currentToConc.increaseSizeTo(size);

  // get fraction volume
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
  {  // soma:
    fractionVolumeER = FRACTIONVOLUME_RoughER;
  }
  else
  {
    fractionVolumeER = FRACTIONVOLUME_SmoothER;
  }

  // initialize data
  Ca_cur[0] = Ca_new[0];
  for (int i = 1; i < size; ++i)
  {
    Ca_new[i] = Ca_new[0];
    Ca_cur[i] = Ca_cur[0];
  }
  // go through each compartments in a branch
  for (int i = 0; i < size; ++i)
  {
    Aii[i] = Aip[i] = Aim[i] = RHS[i] = 0.0;
    currentToConc[i] = getArea(i) * uM_um_cubed_per_pA_msec / getVolume(i);
  }

  // go through different kinds of injected Calcium currents
  //   one of which is the bidirectional current from spine neck
  Array<InjectedCaCurrent>::iterator iiter = injectedCaCurrents.begin();
  Array<InjectedCaCurrent>::iterator iend = injectedCaCurrents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < branchData->size)
      iiter->currentToConc = uM_um_cubed_per_pA_msec / getVolume(iiter->index);
  }

  Aim[0] = Aip[size - 1] = 0;

  if (!isProximalCase0)
  {
    Aip[size - 1] = -getLambda(proximalDimension, dimensions[size - 1]);
  }

  if (isDistalCase1 || isDistalCase2)
  {
    Aim[0] = -getLambda(distalDimensions[0], dimensions[0]);
  }

  for (int i = 1; i < size; i++)
  {
    Aim[i] = -getLambda(dimensions[i - 1], dimensions[i]);
  }

  for (int i = 0; i < size - 1; i++)
  {
    Aip[i] = -getLambda(dimensions[i + 1], dimensions[i]);
  }

  /* FIX */
  if (isDistalCase3)
  {

    // Compute total volume of the junction...
    dyn_var_t volume = getVolume(0);

    // Compute Aij[n] for the junction...one of which goes in Aip[0]...
    if (size == 1)
    {  // branch has only 1 compartment, so get compartment in another branch
      // which is referenced via proximalDimension
      Aip[0] = -getAij(proximalDimension, dimensions[0], volume);
    }
    else
    {
      Aip[0] = -getAij(dimensions[1], dimensions[0], volume);
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
      Aij.push_back(-getAij(distalDimensions[n], dimensions[0], volume));
    }
  }
#ifdef DEBUG_HH
	printDebugHH();
#endif
}

void CaERConcentration::printDebugHH()
{
  unsigned size = branchData->size;
  SegmentDescriptor segmentDescriptor;
  std::cerr << "time| BRANCH [rank, nodeIndex, layerIndex, cptIndex]"
            << "(neuronIdx, brIdx, brOrder, brType)| distalCase(C0,C1,C2,C3) |"
            << "prox(C0,C1,C2) |"
            << "{x,y,z,r, dist2soma, surface_area, volume, length} CaER\n";
  for (int i = 0; i < size; ++i)
  {
    std::cerr << dyn_var_t(getSimulation().getIteration()) *
                     *getSharedMembers().deltaT << " | CAER_BRANCH "
              << " [" << getSimulation().getRank() << "," << getNodeIndex()
              << "," << getIndex() << "," << i << "] "
              << "(" << segmentDescriptor.getNeuronIndex(branchData->key) << ","
              << segmentDescriptor.getBranchIndex(branchData->key) << ","
              << segmentDescriptor.getBranchOrder(branchData->key) << ","
              << segmentDescriptor.getBranchType(branchData->key) 
              << ") |"
              << "(" << isDistalCase0 << "," << isDistalCase1 << "," << isDistalCase2
              << "," << isDistalCase3 << ")|(" << isProximalCase0 << ","
              << isProximalCase1 << "," << isProximalCase2 << ")|"
              << " {" << dimensions[i]->x << "," << dimensions[i]->y << ","
              << dimensions[i]->z << "," << dimensions[i]->r << ","
              << dimensions[i]->dist2soma << "," << dimensions[i]->surface_area
              << "," << dimensions[i]->volume << "," << dimensions[i]->length
              << "} " << Ca_new[i] << " " << std::endl;
  }
}

// Update: RHS[], Aii[]
// Unit: RHS =  [uM/msec]
//       Aii =  [1/msec]
// Thomas algorithm forward step
void CaERConcentration::doForwardSolve()
{
  unsigned size = branchData->size;
	//Find A[ii]i and RHS[ii]  
	//  1. ionic currents 
  for (int i = 0; i < size; i++)
  {
#if CALCIUM_ER_DYNAMICS == FAST_BUFFERING
    Aii[i] = getSharedMembers().bmt - Aim[i] - Aip[i];
    RHS[i] = getSharedMembers().bmt * Ca_cur[i];
#elif CALCIUM_ER_DYNAMICS == REGULAR_BUFFERING
    assert(0);  // need to implement
#endif
    /* * * Sum Currents * * */
    //    Array<ChannelCaCurrents>::iterator iter = channelCaCurrents.begin();
    //    Array<ChannelCaCurrents>::iterator end = channelCaCurrents.end();
    //    for (; iter != end; iter++)
    //    {
    //      RHS[i] -= currentToConc[i] * (*iter->currents)[i];
    //    }
    Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
    Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
    for (; fiter != fend; fiter++)
    {
      // original: RHS[i] -=  (*fiter->fluxes)[i];
      // correct: RHS[i] -=  (*fiter->fluxes)[i] * Vmyo / Ver;
      // equivalent of correct:
      RHS[i] -= (*fiter->fluxes)[i] * FRACTIONVOLUME_CYTO / fractionVolumeER;
    }
  }

  /* FIX */
  if (isDistalCase3)
  {
#if CALCIUM_ER_DYNAMICS == FAST_BUFFERING
    Aii[0] = getSharedMembers().bmt - Aip[0];
    RHS[0] = getSharedMembers().bmt * Ca_cur[0];
#elif CALCIUM_ER_DYNAMICS == REGULAR_BUFFERING
    assert(0);  // need to implement
#endif
    for (int n = 0; n < distalInputs.size(); n++)
    {
      Aii[0] -= Aij[n];
    }
    /* * * Sum Currents * * */
    //    Array<ChannelCaCurrents>::iterator citer = channelCaCurrents.begin();
    //    Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
    //    for (; citer != cend; citer++)
    //    {
    //      RHS[0] -= currentToConc[0] * (*citer->currents)[0];
    //    }
    Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
    Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
    for (; fiter != fend; fiter++)
    {
      // RHS[0] -= (*fiter->fluxes)[0];
      RHS[0] -= (*fiter->fluxes)[0] * FRACTIONVOLUME_CYTO / fractionVolumeER;
    }
  }

  //  Array<ReceptorCaCurrent>::iterator riter = receptorCaCurrents.begin();
  //  Array<ReceptorCaCurrent>::iterator rend = receptorCaCurrents.end();
  //  for (; riter != rend; riter++)
  //  {
  //    int i = riter->index;
  //    RHS[i] -= currentToConc[i] * *(riter->current);
  //  }

  Array<InjectedCaCurrent>::iterator iiter = injectedCaCurrents.begin();
  Array<InjectedCaCurrent>::iterator iend = injectedCaCurrents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < branchData->size)
      RHS[iiter->index] += *(iiter->current) * iiter->currentToConc;
  }

  /* * *  Forward Solve Ax = B * * */
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
  else if (isDistalCase3)
  {
    for (int n = 0; n < distalInputs.size(); n++)
    {
      Aii[0] -= Aij[n] * *distalAips[n] / *distalAiis[n];
      RHS[0] -= Aij[n] * *distalInputs[n] / *distalAiis[n];
    }
  }
  for (int i = 1; i < size; i++)
  {
    Aii[i] -= Aim[i] * Aip[i - 1] / Aii[i - 1];
    RHS[i] -= Aim[i] * RHS[i - 1] / Aii[i - 1];
  }

    // 1.c. HH-like of concentration diffusion
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CAER
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
  
}

// Update; Ca_new[]
// Thomas algorithm backward step
//   - backward substitution on upper triangular matrix
void CaERConcentration::doBackwardSolve()
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
}

dyn_var_t CaERConcentration::getLambda(DimensionStruct* a, DimensionStruct* b)
{
  dyn_var_t radius;
  if (a->dist2soma == 0.0)
  {
    radius = b->r;
  }
  else if (b->dist2soma == 0.0)
    radius = a->r;
  else
    radius = 0.5 * (a->r + b->r);
  // dyn_var_t lengthsq = DISTANCE_SQUARED(a, b);
  // return (getSharedMembers().DCa * radius * radius /
  //        (lengthsq * b->r * b->r)); /* needs fixing */
  dyn_var_t length = std::fabs(b->dist2soma - a->dist2soma);
  //return (getSharedMembers().DCa * radius * radius /
  return (DCa * radius * radius /
          (length * length * b->r * b->r)); /* needs fixing */
}

// GOAL: Get coefficient of Ca(i=0,j=branch-index)
//  DCa * (1/V) * PI * r_(i->j)^2 / (ds_(i->j))
//   V = volume of ER compartment
//   DCa = diffusion constant of CaER
dyn_var_t CaERConcentration::getAij(DimensionStruct* a, DimensionStruct* b,
                                    dyn_var_t V)
{
  dyn_var_t Rb;
  if (a->dist2soma == 0.0)
  {
    Rb = b->r;
  }
  else if (b->dist2soma == 0.0)
    Rb = a->r;
  else
    Rb = 0.5 * (a->r + b->r);
  // return (M_PI * Rb * Rb * getSharedMembers().DCa /
  //        (V * sqrt(DISTANCE_SQUARED(a, b))));
  dyn_var_t length = fabs(b->dist2soma - a->dist2soma);
  //return (M_PI * Rb * Rb * getSharedMembers().DCa / (V * length));
  return (M_PI * Rb * Rb * DCa / (V * length));
}

// void CaERConcentration::setReceptorCaCurrent(
//    const String& CG_direction, const String& CG_component,
//    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
//    Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
//    CG_CaERConcentrationOutAttrPSet* CG_outAttrPset)
//{
//#ifdef DEBUG_ASSERT
//  assert(receptorCaCurrents.size() > 0);
//#endif
//  receptorCaCurrents[receptorCaCurrents.size() - 1].index =
//  CG_inAttrPset->idx;
//}

void CaERConcentration::setInjectedCaCurrent(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset,
    CG_CaERConcentrationOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(injectedCaCurrents.size() > 0);
#endif
  TissueSite& site = CG_inAttrPset->site;
  if (site.r != 0)  // a sphere is provided, i.e. used for current injection
  {  // stimulate a region (any compartments fall within the sphere are
     // affected)
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
              << "Dynamic Cast of CurrentProducer failed in CaERConcentration"
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
  {  // inject at all compartments of one or many branchs meet the condition
    injectedCaCurrents[injectedCaCurrents.size() - 1].index = 0;
    for (int i = 1; i < branchData->size; ++i)
    {
      CaCurrentProducer* CG_CaCurrentProducerPtr =
          dynamic_cast<CaCurrentProducer*>(CG_variable);
      if (CG_CaCurrentProducerPtr == 0)
      {
        std::cerr
            << "Dynamic Cast of CurrentProducer failed in CaERConcentration"
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
  {  // i.e. bi-directional connection (electrical synapse or
     // spineneck-compartment)
    // NOTE: The current component already been assigned via code-generated
    // specified in MDL
    injectedCaCurrents[injectedCaCurrents.size() - 1].index =
        CG_inAttrPset->idx;
  }
}

CaERConcentration::~CaERConcentration() {}

#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CAER
void CaERConcentration::setTargetAttachCaConcentration(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaERConcentrationInAttrPSet* CG_inAttrPset, CG_CaERConcentrationOutAttrPSet* CG_outAttrPset)
{
#ifdef DEBUG_ASSERT
  assert(targetAttachCaConcentration.size() > 0);
#endif
  targetAttachCaConcentration[targetAttachCaConcentration.size() - 1].index = CG_inAttrPset->idx;

}
#endif
