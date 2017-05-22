#include "Lens.h"
#include "CaERConcentrationJunction.h"
#include "CG_CaERConcentrationJunction.h"
#include "rndm.h"
#include "MaxComputeOrder.h"
#include "Branch.h"

//#define DEBUG_HH

#include "SegmentDescriptor.h"
#include "Branch.h"
#include <cmath>

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
  if (_segmentDescriptor.getBranchType(branchData->key) == 0)
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
  currentToConc = getArea() * uM_um_cubed_per_pA_msec / volume;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin(),
    dend = dimensionInputs.end();
  for (; diter != dend; ++diter)
  {
    dyn_var_t Rb;
    dyn_var_t distance;
    if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
    {
      Rb = ((*diter)->r );
#ifdef USE_SOMA_AS_ISOPOTENTIAL
      distance = (*diter)->dist2soma - dimension->r; // SOMA is treated as a point source
#else
      distance = (*diter)->dist2soma ; 
#endif
    }else{
      Rb = 0.5 * ((*diter)->r + dimension->r);
      distance= std::fabs((*diter)->dist2soma - dimension->dist2soma);
    }
    // fAxial.push_back(Pdov * Rb * Rb /
    //                 sqrt(DISTANCE_SQUARED(**diter, *dimension)));
    fAxial.push_back(Pdov * Rb * Rb / distance);
  }
#ifdef DEBUG_HH
  std::cerr << "CaER_JUNCTION (" << dimension->x << "," << dimension->y << ","
    << dimension->z << "," << dimension->r << ")" << std::endl;
#endif
}

void CaERConcentrationJunction::predictJunction(RNG& rng)
{
  assert(getSharedMembers().bmt > 0);
#if CALCIUM_ER_DYNAMICS == FAST_BUFFERING
  float LHS = getSharedMembers().bmt;
  float RHS = getSharedMembers().bmt * Ca_cur ;
#elif CALCIUM_ER_DYNAMICS == REGULAR_BUFFERING
  do something here
#endif

    //  Array<ChannelCaCurrents>::iterator citer = channelCaCurrents.begin();
    //  Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
    //  for (; citer != cend; ++citer)
    //  {
    //    RHS -= currentToConc * (*(citer->currents))[0];
    //  }
    Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
  Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
  for (; fiter != fend; fiter++)
  {
    //RHS -= (*fiter->fluxes)[0];
    RHS -=  (*fiter->fluxes)[0] * FRACTIONVOLUME_CYTO / fractionVolumeER;
  }

  //  Array<dyn_var_t*>::iterator riter = receptorCaCurrents.begin();
  //  Array<dyn_var_t*>::iterator rend = receptorCaCurrents.end();
  //  for (; riter != rend; ++riter)
  //  {
  //    RHS -= currentToConc * **riter;
  //  }

  Array<dyn_var_t*>::iterator iiter, iend;
  iiter = injectedCaCurrents.begin();
  iend = injectedCaCurrents.end();
  for (; iiter != iend; ++iiter)
  {
    RHS += **iiter * currentToConc / getArea();
  }

  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    RHS += (*xiter) * ((**viter) - Ca_cur);
  }

#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CAER
  Array<dyn_var_t*>::iterator titer = targetReversalCaConcentration.begin();
  Array<dyn_var_t*>::iterator tend = targetReversalCaConcentration.end();
  int i = 0;
  for (; titer != tend; ++titer, ++i)
  {
    RHS += *targetInverseTimeCaConcentration[i] * **titer;
  }
#endif

  Ca_new[0] = RHS / LHS;

#ifdef DEBUG_HH
  DimensionStruct* dimension = dimensions[0];  

  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
    << " CaER_JUNCTION PREDICT"
    << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
    << getIndex() << "] "
    << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchIndex(branchData->key) << ","
    << _segmentDescriptor.getBranchOrder(branchData->key) << ") {"
    << dimension->x << "," << dimension->y << "," << dimension->z << ","
    << dimension->r << "} " << Ca_new[0] << std::endl;
#endif
}

void CaERConcentrationJunction::correctJunction(RNG& rng)
{
#if CALCIUM_ER_DYNAMICS == FAST_BUFFERING
  assert(getSharedMembers().bmt > 0);
  float LHS = getSharedMembers().bmt;
  float RHS = getSharedMembers().bmt * Ca_cur;
#elif CALCIUM_ER_DYNAMICS == REGULAR_BUFFERING
  do something here
#endif

    //  Array<ChannelCaCurrents>::iterator citer = channelCaCurrents.begin();
    //  Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
    //  for (; citer != cend; ++citer)
    //  {
    //    RHS -= currentToConc * (*(citer->currents))[0];
    //  }
    Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
  Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
  for (; fiter != fend; fiter++)
  {
    //RHS -= (*fiter->fluxes)[0];
    RHS -=  (*fiter->fluxes)[0] * FRACTIONVOLUME_CYTO / fractionVolumeER;
  }

  //  Array<dyn_var_t*>::iterator riter = receptorCaCurrents.begin();
  //  Array<dyn_var_t*>::iterator rend = receptorCaCurrents.end();
  //  for (; riter != rend; ++riter)
  //  {
  //    RHS -= currentToConc * **riter;
  //  }

  Array<dyn_var_t*>::iterator iiter, iend;
  iiter = injectedCaCurrents.begin();
  iend = injectedCaCurrents.end();
  for (; iiter != iend; ++iiter)
  {
    RHS += **iiter * currentToConc / getArea();
  }

  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    LHS += (*xiter);
    RHS += (*xiter) * (**viter);
  }

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

  Ca_new[0] = RHS / LHS;

  // This is the swap phase
  Ca_cur = Ca_new[0] = 2.0 * Ca_new[0] - Ca_cur;

#ifdef DEBUG_HH
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
  int c = 0;

  for (viter = CaConcentrationInputs.begin(); viter != vend; ++viter, ++diter)
  {
    std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
      << " CaER_JCT_INPUT_" << c++ << " [" << getSimulation().getRank()
      << "," << getNodeIndex() << "," << getIndex() << "] "
      << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
      << _segmentDescriptor.getBranchIndex(branchData->key) << ","
      << _segmentDescriptor.getBranchOrder(branchData->key) << ","
      << _segmentDescriptor.getComputeOrder(branchData->key) << ") {"
      << (*diter)->x << "," << (*diter)->y << "," << (*diter)->z << ","
      //<< (*diter)->r << "} " << DISTANCE_SQUARED(*(*diter),
      //*dimension)
      << (*diter)->r << "} "
      << (((*diter))->dist2soma - dimension->dist2soma) << " "
      << *(*viter) << std::endl;
  }
#endif
}

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
