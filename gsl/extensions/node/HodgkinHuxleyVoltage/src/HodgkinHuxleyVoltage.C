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
#include "HodgkinHuxleyVoltage.h"
#include "CG_HodgkinHuxleyVoltage.h"
#include "rndm.h"
#include "GridLayerDescriptor.h"
#include "MaxComputeOrder.h"

#define DISTANCE_SQUARED(a, b)               \
  ((((a)->x - (b)->x) * ((a)->x - (b)->x)) + \
   (((a)->y - (b)->y) * ((a)->y - (b)->y)) + \
   (((a)->z - (b)->z) * ((a)->z - (b)->z)))

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
  for (int i = 0; !atSite && i < dimensions.size(); ++i)
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


// update: V(t+dt) = 2 * V(t+dt/2) - V(t)
// second-step (final step) in Crank-Nicolson method
void HodgkinHuxleyVoltage::finish(RNG& rng)
{
  unsigned size = branchData->size;
#ifdef DEBUG_HH
  SegmentDescriptor segmentDescriptor;
  for (int i = 0; i < size; ++i)
  {
    std::cerr << dyn_var_t(getSimulation().getIteration()) *
                     *getSharedMembers().deltaT << " BRANCH"
              << " [" << getSimulation().getRank() << "," << getNodeIndex()
              << "," << getIndex() << "," << i << "] "
              << "(" << segmentDescriptor.getNeuronIndex(branchData->key) << ","
              << segmentDescriptor.getBranchIndex(branchData->key) << ","
              << segmentDescriptor.getBranchOrder(branchData->key) << ") |"
              << isDistalCase0 << "|" << isDistalCase1 << "|" << isDistalCase2
              << "|" << isDistalCase3 << "|" << isProximalCase0 << "|"
              << isProximalCase1 << "|" << isProximalCase2 << "|"
              << " {" << dimensions[i]->x << "," << dimensions[i]->y << ","
              << dimensions[i]->z << "," << dimensions[i]->r << "} " << Vnew[i]
              << " " << std::endl;
  }
#endif
  for (int i = 0; i < size; ++i)
  {
    Vcur[i] = Vnew[i] = 2.0 * Vnew[i] - Vcur[i];
#ifdef DEBUG_ASSERT
    assert(Vnew[i] == Vnew[i]);  // making sure Vnew[i] is not NaN
#endif
  }
}

// membrane surface area of the compartment based on its index 'i'
dyn_var_t HodgkinHuxleyVoltage::getArea(int i)  // TUAN: check ok
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
void HodgkinHuxleyVoltage::initializeCompartmentData(RNG& rng)  // TUAN: checked
                                                                // ok
{
  // for a given computing process:
  //  here all the data in vector-form are initialized to
  //  the same size of the number of compartments in a branch (i.e. branchData)
  unsigned size = branchData->size;  //# of compartments
  SegmentDescriptor segmentDescriptor;
  computeOrder = segmentDescriptor.getComputeOrder(branchData->key);
#ifdef DEBUG_ASSERT
  if (isProximalCase2) assert(computeOrder == 0);
  if (isDistalCase2) assert(computeOrder == MAX_COMPUTE_ORDER);
  assert(dimensions.size() == size);
  assert(Vnew.size() == size);
  assert(distalDimensions.size() == distalInputs.size());
#endif

  // allocate data
  if (Vcur.size() != size) Vcur.increaseSizeTo(size);
  if (Aii.size() != size) Aii.increaseSizeTo(size);
  if (Aip.size() != size) Aip.increaseSizeTo(size);
  if (Aim.size() != size) Aim.increaseSizeTo(size);
  if (RHS.size() != size) RHS.increaseSizeTo(size);

  // initialize data
  Vcur[0] = Vnew[0];
  for (int i = 1; i < size; ++i)
  {
    Vnew[i] = Vnew[0];
    Vcur[i] = Vcur[0];
  }
  for (int i = 0; i < size; ++i)
  {
    Aii[i] = Aip[i] = Aim[i] = RHS[i] = 0.0;
  }

  // get surface area of the compartment and put into InjectedCurrent structure
  Array<InjectedCurrent>::iterator iiter = injectedCurrents.begin();
  Array<InjectedCurrent>::iterator iend = injectedCurrents.end();
  for (; iiter != iend; iiter++)
  {
    if (iiter->index < branchData->size) iiter->area = getArea(iiter->index);
  }

  Aim[0] = Aip[size - 1] = 0;
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (!isProximalCase0)
  {
    Aip[size - 1] =
        -getLambda(proximalDimension, dimensions[size - 1]);  // [nS/um^2]
  }
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (isDistalCase1 || isDistalCase2)
  {
    Aim[0] = -getLambda(distalDimensions[0], dimensions[0]);
  }
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  for (int i = 1; i < size; i++)
  {
    Aim[i] = -getLambda(dimensions[i - 1], dimensions[i]);
  }
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  for (int i = 0; i < size - 1; i++)
  {
    Aip[i] = -getLambda(dimensions[i + 1], dimensions[i]);
  }

  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (isDistalCase3)
  {
  //IMPORTANT CHANGE:
  // Unlike the original approach
  //   which doesn't have a compartment for the implicit branching
  // The branch now has
  //at least 2: one compartment as implicit branching point + one as regular
  //    compartment-zero as implicit branching compartment
  //    compartment-1th and above as normal
	  
    // Compute total area of the junction...
    dyn_var_t area = getArea(0);

    // Compute Aij[n] for the junction...one of which goes in Aip[0]...
    /*if (size == 1)
    {
      Aip[0] = -getAij(proximalDimension, dimensions[0], area);
    }
    else
    {
      Aip[0] = -getAij(dimensions[1], dimensions[0], area);
    }*/
    Aip[0] = -getAij(dimensions[1], dimensions[0], area);
    for (int n = 0; n < distalDimensions.size(); n++)
    {
      Aij.push_back(-getAij(distalDimensions[n], dimensions[0], area));
    }
  }
  if (getSharedMembers().deltaT)
  {
    cmt = 2.0 * Cm / *(getSharedMembers().deltaT);  // [pF/(um^2 . ms)]
  }
}

// Update: RHS[], Aii[]
// Convert to upper triangular matrix  
// Thomas algorithm forward step 
void HodgkinHuxleyVoltage::doForwardSolve()
{
  unsigned size = branchData->size;
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  for (int i = 0; i < size; i++)  // for each compartment on that branch
  {
    Aii[i] = cmt - Aim[i] - Aip[i] + gLeak;
    RHS[i] = cmt * Vcur[i] + gLeak * getSharedMembers().E_leak;
    /* * * Sum Currents * * */
    // loop through different kinds of currents (Kv, Nav1.6, ...)
    Array<ChannelCurrents>::iterator iter = channelCurrents.begin();
    Array<ChannelCurrents>::iterator end = channelCurrents.end();
    for (int k = 0; iter != end; iter++, ++k)
    {
      ShallowArray<dyn_var_t>* conductances = iter->conductances;
      // at each current type, there is an array of currents of that type,
      //...each current element flows into one compartment
      RHS[i] +=
          (*conductances)[i] *
          (*(iter->reversalPotentials))[(iter->reversalPotentials->size() == 1)
                                            ? 0
                                            : i];
      Aii[i] += (*conductances)[i];
    }
  }
  // JMW 07/10/2009 CHECKED AND LOOKS RIGHT
  if (isDistalCase3)
  {
    Aii[0] = cmt - Aip[0] + gLeak;                               //[nS/um^2]
    RHS[0] = cmt * Vcur[0] + gLeak * getSharedMembers().E_leak;  //[pA/um^2]
    for (int n = 0; n < distalInputs.size(); n++)
    {
      Aii[0] -= Aij[n];
    }
    /* * * Sum Currents * * */
    Array<ChannelCurrents>::iterator citer = channelCurrents.begin();
    Array<ChannelCurrents>::iterator cend = channelCurrents.end();
    for (; citer != cend; citer++)
    {
      ShallowArray<dyn_var_t>* conductances = citer->conductances;  // [nS/um^2]
      RHS[0] +=
          (*conductances)[0] * (*(citer->reversalPotentials))[0];  // [pA/um^2]
      Aii[0] += (*conductances)[0];
    }
  }

  Array<ReceptorCurrent>::iterator riter = receptorCurrents.begin();
  Array<ReceptorCurrent>::iterator rend = receptorCurrents.end();
  for (; riter != rend; riter++)
  {
    int i = riter->index;
    RHS[i] += *(riter->conductance) * *(riter->reversalPotential);
    Aii[i] += *(riter->conductance);
  }

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
      RHS[iiter->index] += *(iiter->current) / iiter->area;
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
}  // end doForwardSolve

// Update: Vnew[]
// Thomas algorithm backward step 
//   - backward substitution on upper triangular matrix
void HodgkinHuxleyVoltage::doBackwardSolve()
{
  unsigned size = branchData->size;
  if (isProximalCase0)
  {
    Vnew[size - 1] = RHS[size - 1] / Aii[size - 1];
  }
  else
  {
    Vnew[size - 1] =
        (RHS[size - 1] - Aip[size - 1] * *proximalVoltage) / Aii[size - 1];
  }
  for (int i = size - 2; i >= 0; i--)
  {
    Vnew[i] = (RHS[i] - Aip[i] * Vnew[i + 1]) / Aii[i];
  }
}  // end doBackwardSolve

// unit: [nS/um^2]
// GOA: get 'lambda' term between two adjacent compartments
dyn_var_t HodgkinHuxleyVoltage::getLambda(DimensionStruct* a,
                                          DimensionStruct* b)
{
  dyn_var_t radius = 0.5 * (a->r + b->r);  // radius_middle ()
  // dyn_var_t lengthsq = DISTANCE_SQUARED(a, b);
  //return (radius * radius /
  //        (2.0 * getSharedMembers().Ra * lengthsq * b->r)); /* needs fixing */
  dyn_var_t length = abs(b->dist2soma - a->dist2soma);
  return (radius * radius /
          (2.0 * getSharedMembers().Ra * length * length * b->r)); /* needs fixing */
}

// GOAL get the Aij[]
//  A = surface_area
dyn_var_t HodgkinHuxleyVoltage::getAij(DimensionStruct* a, DimensionStruct* b,
                                       dyn_var_t A)
{
  dyn_var_t Rb = 0.5 * (a->r + b->r);
  // dyn_var_t length = sqrt(DISTANCE_SQUARED(a, b);
  dyn_var_t length = abs(b->dist2soma - a->dist2soma);
  return (M_PI * Rb * Rb /
          (A * getSharedMembers().Ra * length));
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
  {//stimulation purpose
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
  else if (CG_inAttrPset->idx < 0)  //??? TUAN : which condition is this
  {//inject at all compartments???
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
    injectedCurrents[injectedCurrents.size() - 1].index = CG_inAttrPset->idx;
  }
}

HodgkinHuxleyVoltage::~HodgkinHuxleyVoltage() {}
