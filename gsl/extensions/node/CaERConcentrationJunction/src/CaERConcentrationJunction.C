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
#include "CaERConcentrationJunction.h"
#include "CG_CaERConcentrationJunction.h"
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
#include "StringUtils.h"
#include "Params.h"

#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

// NOTE: value = 1e6/(zCa*Farad)
// zCa = valence of Ca2+
// Farad = Faraday's constant
#define uM_um_cubed_per_pA_msec 5.18213484752067

SegmentDescriptor CaERConcentrationJunction::_segmentDescriptor;

#if CALCIUM_ER_DYNAMICS == FAST_BUFFERING
#define DCa (getSharedMembers().DCaeff)
#else
#define DCa (getSharedMembers().DCa)
#endif

// Get endoplasmic reticular surface area at the compartment i-th
// Check if smooth or rough ER
dyn_var_t CaERConcentrationJunction::getArea()  // Tuan: check ok
{
  dyn_var_t area = 0.0;
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
  {
#if defined (USE_SOMA_AS_POINT)
    area = 1.0 * FRACTION_SURFACEAREA_SmoothER; // [um^2]
#else
    area = dimensions[0]->surface_area * FRACTION_SURFACEAREA_SmoothER;
#endif
  }
  else
  {
    area = dimensions[0]->surface_area * FRACTION_SURFACEAREA_RoughER;
  }
  return area;
}

// Get endoplasmic reticulum volume at the compartment i-th
// Check if smooth or rough ER
dyn_var_t CaERConcentrationJunction::getVolume()  // Tuan: check ok
{
  dyn_var_t volume = 0.0;
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
  {
#if defined (USE_SOMA_AS_POINT)
    volume = 1.0 * FRACTIONVOLUME_SmoothER; // [um^3]
#else
    volume = dimensions[0]->volume * FRACTIONVOLUME_SmoothER;
#endif
  }
  else
  {
    volume = dimensions[0]->volume * FRACTIONVOLUME_RoughER;
  }
  return volume;
}

void CaERConcentrationJunction::initializeJunction(RNG& rng)
{  // explicit junction (which can be soma (with branches are axon/dendrite
  // trees)
  // or a cut point junction
  // or a branching point junction with 3 or more branches (one from main, 2+ for
  // children
  // branches))
#ifdef DEBUG_ASSERT
  assert(Ca_new.size() == 1);
  assert(dimensions.size() == 1);
#endif

  //get fraction volume
  if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
  {  // soma:
    fractionVolumeER = FRACTIONVOLUME_RoughER;
  }else{ 
    fractionVolumeER = FRACTIONVOLUME_SmoothER;
  }

  Ca_cur = Ca_new[0];
  // So, one explicit junction is composed of one compartment
  // which can be explicit cut-point junction or
  //              explicit branching-point junction
  DimensionStruct* dimension = dimensions[0];

  Array<DimensionStruct*>::iterator iter = dimensionInputs.begin(),
    end = dimensionInputs.end();

  volume = getVolume();

  float Pdov = M_PI * DCa / volume;
  currentDensityToConc = getArea() * uM_um_cubed_per_pA_msec / volume;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin(),
    dend = dimensionInputs.end();
  for (; diter != dend; ++diter)
  {
    dyn_var_t Rb;
    dyn_var_t distance;
    if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
    {
      Rb = ((*diter)->r );
#ifdef USE_SCALING_NECK_FROM_SOMA
      //TEST 
      Rb /= SCALING_NECK_FROM_SOMA;
      //END TEST
#endif

#ifdef USE_SOMA_AS_ISOPOTENTIAL
      distance = (*diter)->dist2soma - dimension->r; // SOMA is treated as a point source
#else
      distance = (*diter)->dist2soma ; 
#ifdef USE_STRETCH_SOMA_RADIUS
      //TEST 
      distance += STRETCH_SOMA_WITH;
      //  distance += 50.0;//TUAN TESTING - make soma longer
      //distance = std::fabs(b->r + a->dist2soma);
      //END TEST
#endif
#endif
    }else{
      Rb = 0.5 * ((*diter)->r + dimension->r);
      distance= std::fabs((*diter)->dist2soma - dimension->dist2soma);
    }
    Rb *= sqrt(FRACTION_CROSS_SECTIONALAREA_ER);
    fAxial.push_back(Pdov * Rb * Rb / distance);
  }
#ifdef DEBUG_HH
  std::cerr << "CaER_JUNCTION (" << dimension->x << "," << dimension->y << ","
    << dimension->z << "," << dimension->r << ")" << std::endl;
#endif
}

// GOAL: predict Canew[0] at offset time (t+dt/2) - Crank-Nicolson predictor-corrector scheme
//    using Ca_branch(t) and Canew[0](t)
void CaERConcentrationJunction::predictJunction(RNG& rng)
{
  //element-1
  float LHS = getSharedMembers().bmt; // [1/ms]
  float RHS = getSharedMembers().bmt * Ca_cur ; // [uM/ms]


  //element-2 
  // no integrated 'extrusion' 

  /* * * Sum Currents * * */
  //  Array<ChannelCaCurrents>::iterator citer = channelCaCurrents.begin();
  //  Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
  //  for (; citer != cend; ++citer)
  //  {
  //    RHS -= currentDensityToConc * (*(citer->currents))[0];
  //  }
  // 1.a. those produces J(t)  [uM/ms^2]
  Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
  Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
  for (; fiter != fend; fiter++)
  {
    //IMPORTANT - flux toward volume cyto (so gain for cyto; but loss for ER)
    RHS -=  (*fiter->fluxes)[0] * FRACTIONVOLUME_CYTO / fractionVolumeER;
  }

  //  Array<dyn_var_t*>::iterator riter = receptorCaCurrents.begin();
  //  Array<dyn_var_t*>::iterator rend = receptorCaCurrents.end();
  //  for (; riter != rend; ++riter)
  //  {
  //    RHS -= currentDensityToConc * **riter;
  //  }

  //  4. injected currents  [pA]
  Array<dyn_var_t*>::iterator iiter, iend;
  iiter = injectedCaCurrents.begin();
  iend = injectedCaCurrents.end();
  for (; iiter != iend; ++iiter)
  {
    RHS += **iiter * currentDensityToConc / getArea();
  }

  // 5. Concentration loss due to passive diffusion to adjacent compartments
  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    RHS += (*xiter) * ((**viter) - Ca_cur);
  }

  // 6. Concentration via spine neck 
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CAER
  Array<dyn_var_t*>::iterator titer = targetReversalCaConcentration.begin();
  Array<dyn_var_t*>::iterator tend = targetReversalCaConcentration.end();
  int i = 0;
  for (; titer != tend; ++titer, ++i)
  {
    RHS += *targetInverseTimeCaConcentration[i] * **titer;
  }
#endif

  Ca_new[0] = RHS / LHS;  // estimate at (t+dt/2)

#ifdef DEBUG_HH
  DimensionStruct* dimension = dimensions[0];  

  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
    << " CaER_JUNCTION PREDICT"
    << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
    << getIndex() << "] "
    << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchOrder(branchData->key) << ") {"
    << dimension->x << "," << dimension->y << "," 
    << dimension->z << ","
    << dimension->r << "} " << Ca_new[0] << std::endl;
#endif
}

// GOAL: do 2 things:
//  1. correct Canew[0] at (t+dt/2) 
//  2. update Cacur, and Canew[0] at (t+dt) 
void CaERConcentrationJunction::correctJunction(RNG& rng)
{
  //element-1
  float LHS = getSharedMembers().bmt;   // [1/ms]
  float RHS = getSharedMembers().bmt * Ca_cur;  // [uM/ms]

  //element-2 
  // no integrated 'extrusion' 
  

  /* * * Sum Currents * * */
  // 1.a. those produces I(t)  [pA/um^2]
  //  Array<ChannelCaCurrents>::iterator citer = channelCaCurrents.begin();
  //  Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
  //  for (; citer != cend; ++citer)
  //  {
  //    RHS -= currentDensityToConc * (*(citer->currents))[0];
  //  }
  // 1.a. those produces J(t)  [uM/ms^2]
  Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
  Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
  for (; fiter != fend; fiter++)
  {
    RHS -=  (*fiter->fluxes)[0] * FRACTIONVOLUME_CYTO / fractionVolumeER;
  }

  //  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  //  Array<dyn_var_t*>::iterator riter = receptorCaCurrents.begin();
  //  Array<dyn_var_t*>::iterator rend = receptorCaCurrents.end();
  //  for (; riter != rend; ++riter)
  //  {
  //    RHS -= currentDensityToConc * **riter;
  //  }

  //  4. injected currents  [pA]
  Array<dyn_var_t*>::iterator iiter, iend;
  iiter = injectedCaCurrents.begin();
  iend = injectedCaCurrents.end();
  for (; iiter != iend; ++iiter)
  {
    RHS += **iiter * currentDensityToConc / getArea();
  }

  // 5. Concentration loss due to passive diffusion to adjacent compartments
  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    LHS += (*xiter);
    RHS += (*xiter) * (**viter);
  }

  // 6. Concentration via spine neck 
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CAER
  Array<dyn_var_t*>::iterator titer = targetReversalCaConcentration.begin();
  Array<dyn_var_t*>::iterator tend = targetReversalCaConcentration.end();
  int i = 0;
  for (; titer != tend; ++titer, ++i)
  {
    RHS += *targetInverseTimeCaConcentration[i] * **titer;
    LHS += *targetInverseTimeCaConcentration[i] ;
  }
#endif

  Ca_new[0] = RHS / LHS; //corrected value at (t+dt/2)

  // This is the swap phase
  Ca_cur = Ca_new[0] = 2.0 * Ca_new[0] - Ca_cur;  //value at (t+dt)

#ifdef DEBUG_HH
  printDebugHH();
#endif
}

void CaERConcentrationJunction::printDebugHH(std::string phase)
{
  std::cerr << "step,time|" << phase << " [rank,nodeIdx,instanceIdx] " <<
    "(neuronIdx,branchIdx,brchOrder){x,y,z,r | dist2soma,surfarea,volume,len} Vm" << std::endl;
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
    << " CaER_JUNCTION CORRECT"
    << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
    << getIndex() << "] "
    << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchOrder(branchData->key) << ") {"
    << dimension->x << "," << dimension->y << "," << dimension->z << ","
    << dimension->r << "} " << Ca_new[0] << std::endl;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin();
  Array<dyn_var_t*>::iterator vend = CaConcentrationInputs.end();
  int c = -1;

  std::cerr << "JCT_INPUT_i " <<
    "(neuronIdx,branchIdx,brchOrder, brType, COMPUTEORDER){x,y,z,r | dist2soma,surfarea,volume,len} Vm" << std::endl;
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (viter = CaConcentrationInputs.begin(); viter != vend; ++viter, ++diter)
  {
    std::cerr << " CaER_JCT_INPUT_" << c++ 
      << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
      << std::setw(2) << _segmentDescriptor.getBranchIndex(branchData->key) << ","
      << _segmentDescriptor.getBranchOrder(branchData->key) << ","
      //<< _segmentDescriptor.getBranchType(branchDataInputs[c]->key) << ","
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
bool CaERConcentrationJunction::checkSite(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_CaERConcentrationJunctionInAttrPSet* CG_inAttrPset,
    CG_CaERConcentrationJunctionOutAttrPSet* CG_outAttrPset)
{
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  TissueSite& site = CG_inAttrPset->site;
  bool rval = (site.r == 0);
  if (!rval) rval = ((site.r * site.r) >= DISTANCE_SQUARED(site, *dimension));
  return rval;
}

bool CaERConcentrationJunction::confirmUniqueDeltaT(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_CaERConcentrationJunctionInAttrPSet* CG_inAttrPset,
    CG_CaERConcentrationJunctionOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}

CaERConcentrationJunction::~CaERConcentrationJunction() {}
