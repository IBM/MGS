// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "CaConcentrationJunction.h"
#include "CG_CaConcentrationJunction.h"
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

SegmentDescriptor CaConcentrationJunction::_segmentDescriptor;

#if CALCIUM_CYTO_DYNAMICS == FAST_BUFFERING
#define DCa (getSharedMembers().DCaeff)
#else
#define DCa (getSharedMembers().DCa)
#endif

// Get cytoplasmic surface area at the compartment i-th 
dyn_var_t CaConcentrationJunction::getArea() // Tuan: check ok
{
  dyn_var_t area= 0.0;
#if defined (USE_SOMA_AS_POINT)
  area = 1.0 * FRACTION_SURFACEAREA_CYTO; // [um^2]
#else
  area = dimensions[0]->surface_area * FRACTION_SURFACEAREA_CYTO;
#endif
  return area;
}

// Get cytoplasmic volume at the compartment i-th 
dyn_var_t CaConcentrationJunction::getVolume() // Tuan: check ok
{
  dyn_var_t volume = 0.0;
#if defined (USE_SOMA_AS_POINT)
  volume = 1.0 * FRACTIONVOLUME_CYTO; // [um^3]
#else
  volume = dimensions[0]->volume * FRACTIONVOLUME_CYTO;
#endif
  return volume;
}

void CaConcentrationJunction::initializeJunction(RNG& rng)
{// explicit junction (which can be soma (with branches are axon/dendrite
  // trees)
  // or a cut point junction 
  // or a branching point junction with 3 or more branches (one from main, 2+ for children
  // branches))
  assert(Ca_new.size() == 1);
  assert(dimensions.size() == 1);

  Ca_cur = Ca_new[0];
#ifdef MICRODOMAIN_CALCIUM
  unsigned int ii=0;
  for (ii = 0; ii < Ca_microdomain.size(); ii++)
  {
    Ca_microdomain[ii] = Ca_cur;
    Ca_microdomain_cur[ii] = Ca_cur;
  }
#endif
  // So, one explicit junction is composed of one compartment 
  // which can be explicit cut-point junction or
  //              explicit branching-point junction
  DimensionStruct* dimension = dimensions[0];  

  volume = getVolume();

  float Pdov = M_PI * DCa / volume;
#ifdef USE_SUBSHELL_FOR_SOMA
  if ((_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA) &&
      dimension->r > THRESHOLD_SIZE_R_SOMA // to avoid the confusing of spine head
      )//TUAN TODO: consider fixing this
  {
    //for soma: due to large volume, we scale up the [Ca2+]
    // shell volume = 4/3 * pi * (rsoma^3 - (rsoma-d)^3)
    // with d = shell depth
    // RATIO = somaVolume / shellVolume;
    // currentDensityToConc = getArea() * uM_um_cubed_per_pA_msec / volume * RATIO ;
    // TUAN TODO - 
    dyn_var_t d = 1.0; //[um] - shell depth (default) 
    //dyn_var_t d = 0.5; //[um]  
    //dyn_var_t d = 0.2; //[um]  
    if (GlobalNTS::shellDepth > 0.0)
      d = GlobalNTS::shellDepth;
    dyn_var_t shellVolume = 4.0 / 3.0 * M_PI * 
      (pow(dimension->r,3) - pow(dimension->r - d, 3)) * FRACTIONVOLUME_CYTO;
    currentDensityToConc = getArea() * uM_um_cubed_per_pA_msec / shellVolume;
    //std::cerr << "Cyto total vol: " << volume << "; shell volume: " << shellVolume << std::endl;
    
    Pdov = M_PI * DCa / shellVolume;

  }
  else
    currentDensityToConc = getArea() * uM_um_cubed_per_pA_msec / volume;
#else
  currentDensityToConc = getArea() * uM_um_cubed_per_pA_msec / volume;
#endif
#ifdef MICRODOMAIN_CALCIUM
    //for (unsigned int ii=0; ii < microdomainNames.size(); ++ii)
    //{
    //  currentDensityToConc_microdomain[ii] = getArea() * uM_um_cubed_per_pA_msec / volume_microdomain[ii];
    //}
#endif

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin(),
                                    dend = dimensionInputs.end();
  for (; diter != dend; ++diter)
  {
    //NOTE: if the junction is the SOMA, we should not use the radius of the SOMA
    //      in calculating the cross-sectional area
    dyn_var_t Rb;
    dyn_var_t distance;
    if (_segmentDescriptor.getBranchType(branchData->key) == Branch::_SOMA)
    {
      //Rb = ((*diter)->r ) * 1.5;  //scaling factor 1.5 means the bigger interface with soma
      //  NOTE: should be applied for Axon hillock only
      Rb = ((*diter)->r );
#ifdef USE_SCALING_NECK_FROM_SOMA
      //TEST 
      Rb /= SCALING_NECK_FROM_SOMA_WITH;
      //END TEST
#endif

#ifdef USE_SOMA_AS_ISOPOTENTIAL
      distance = (*diter)->dist2soma - dimension->r; // SOMA is treated as a point source
#else
      //distance= std::fabs((*diter)->dist2soma + dimension->r );
      distance = (*diter)->dist2soma; //NOTE: The dist2soma of the first compartment stemming
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
      if (distance <= 0)
        std::cerr << "distance = " << distance << ": " << (*diter)->dist2soma << ","<< dimension->r << std::endl;
      assert(distance > 0);
    }else{
#ifdef NEW_RADIUS_CALCULATION_JUNCTION
      Rb = ((*diter)->r); //the small diameter of the branch means small current pass to it
#else
      Rb = 0.5 * ((*diter)->r + dimension->r);
#endif
      distance= std::fabs((*diter)->dist2soma - dimension->dist2soma);
      assert(distance > 0);
    }
    fAxial.push_back(Pdov * Rb * Rb / distance );
  }
#ifdef DEBUG_HH
  std::cerr << "CA_JUNCTION (" << dimension->x << "," << dimension->y << ","
    << dimension->z << "," << dimension->r << ")" << std::endl;
#endif
}

// GOAL: predict Canew[0] at offset time (t+dt/2) - Crank-Nicolson predictor-corrector scheme
//    using Ca_branch(t) and Canew[0](t)
void CaConcentrationJunction::predictJunction(RNG& rng)
{
  //element-1
  double LHS = getSharedMembers().bmt; // [1/ms]
  double RHS = getSharedMembers().bmt * Ca_cur ;  // [uM/ms]

#ifdef MICRODOMAIN_CALCIUM
  //predict at time (t + dt/2)
  if (microdomainNames.size() > 0)
  {
    updateMicrodomains(LHS, RHS);
    updateMicrodomains_Ca();
  }
#endif

  //element-2 
  // no integrated 'extrusion' --> use explicit PMCA
  
  /* * * Sum Currents * * */
  // 1.a. those produces I(t)  [pA/um^2]
  Array<ChannelCaCurrents>::iterator citer;
  Array<ChannelCaCurrents>::iterator cend ;
  citer = channelCaCurrents.begin();
  cend = channelCaCurrents.end();
  for (; citer != cend; ++citer)
  {
    RHS -= currentDensityToConc * (*(citer->currents))[0];
  }

  // 1.b. those produces J(t)  [uM/ms^2]
  Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
  Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
  for (; fiter != fend; fiter++)
  {
    RHS +=  (*fiter->fluxes)[0];
  }

  //  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<dyn_var_t*>::iterator iter = receptorCaCurrents.begin();
  Array<dyn_var_t*>::iterator end = receptorCaCurrents.end();
  for (; iter != end; ++iter)
  {
    RHS -= currentDensityToConc * **iter;
  }

  //  3. synapse receptor currents using GHK type equations 
  //  NOTE: Not available
  //{
  //  Array<ReceptorCaCurrentsGHK>::iterator riter = receptorCaCurrentsGHK.begin();
  //  Array<ReceptorCaCurrentsGHK>::iterator rend = receptorCaCurrentsGHK.end();
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

  //  4. injected currents  [pA]
  iter = injectedCaCurrents.begin();
  end = injectedCaCurrents.end();
  for (; iter != end; ++iter)
  {
    RHS += **iter * currentDensityToConc / getArea();
  }

  // 5. Concentration loss due to passive diffusion to adjacent compartments
  Array<dyn_var_t>::iterator xiter = fAxial.begin(), xend = fAxial.end();
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (; xiter != xend; ++xiter, ++viter)
  {
    RHS += (*xiter) * ((**viter) - Ca_cur);
  }

  // 6. Concentration via spine neck 
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  Array<dyn_var_t*>::iterator titer = targetReversalCaConcentration.begin();
  Array<dyn_var_t*>::iterator tend = targetReversalCaConcentration.end();
  int i = 0;
  for (; titer != tend; ++titer, ++i)
  {
    RHS += *targetInverseTimeCaConcentration[i] * **titer;
  }
#endif

  Ca_new[0] = RHS / LHS;  //estimate at (t+dt/2)

#ifdef DEBUG_HH
  std::cerr << getSimulation().getIteration() * *getSharedMembers().deltaT
            << " CA_JUNCTION PREDICT"
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
            << Ca_new[0] << std::endl;
#endif
}

// GOAL: do 2 things:
//  1. correct Canew[0] at (t+dt/2) 
//  2. update Cacur, and Canew[0] at (t+dt) 
void CaConcentrationJunction::correctJunction(RNG& rng)
{
  //element-1
  double LHS = getSharedMembers().bmt; // [1/ms]
  double RHS = getSharedMembers().bmt * Ca_cur;  // [uM/ms]

#ifdef MICRODOMAIN_CALCIUM
  //correct at time (t + dt/2)
  if (microdomainNames.size() > 0)
    updateMicrodomains(LHS, RHS);
#endif

  //element-2 
  // no integrated 'extrusion' --> use explicit PMCA
  

  /* * * Sum Currents * * */
  // 1.a. those produces I(t)  [pA/um^2]
  Array<ChannelCaCurrents>::iterator citer = channelCaCurrents.begin();
  Array<ChannelCaCurrents>::iterator cend = channelCaCurrents.end();
  for (; citer != cend; ++citer)
  {
    RHS -= currentDensityToConc * (*(citer->currents))[0];
  }

  // 1.a. those produces J(t)  [uM/ms^2]
  Array<ChannelCaFluxes>::iterator fiter = channelCaFluxes.begin();
  Array<ChannelCaFluxes>::iterator fend = channelCaFluxes.end();
  for (; fiter != fend; fiter++)
  {
    RHS +=  (*fiter->fluxes)[0];
  }

  //  2. synapse receptor currents using Hodgkin-Huxley type equations (gV, gErev)
  Array<dyn_var_t*>::iterator iter = receptorCaCurrents.begin();
  Array<dyn_var_t*>::iterator end = receptorCaCurrents.end();
  for (; iter != end; ++iter)
  {
    RHS -= currentDensityToConc * **iter;
  }

  //  3. synapse receptor currents using GHK type equations 
  //  NOTE: Not available
  //{
  //  Array<ReceptorCaCurrentsGHK>::iterator riter = receptorCaCurrentsGHK.begin();
  //  Array<ReceptorCaCurrentsGHK>::iterator rend = receptorCaCurrentsGHK.end();
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

  //  4. injected currents  [pA]
  iter = injectedCaCurrents.begin();
  end = injectedCaCurrents.end();
  for (; iter != end; ++iter)
  {
    RHS += **iter * currentDensityToConc / getArea();
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
#ifdef CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO
  Array<dyn_var_t*>::iterator titer = targetReversalCaConcentration.begin();
  Array<dyn_var_t*>::iterator tend = targetReversalCaConcentration.end();
  Array<dyn_var_t*>::iterator tviter = targetInverseTimeCaConcentration.begin();
  for (; titer != tend; ++titer, ++tviter)
  {
    RHS += **tviter * **titer;
    LHS += **tviter ;
  }
#endif

  Ca_new[0] = RHS / LHS; //corrected value at (t + dt/2)
#ifdef MICRODOMAIN_CALCIUM
  //correct at time (t + dt/2)
  if (microdomainNames.size() > 0)
    updateMicrodomains_Ca();
#endif

  // This is the swap phase
  Ca_cur = Ca_new[0] = 2.0 * Ca_new[0] - Ca_cur;//value at (t+dt)
#ifdef MICRODOMAIN_CALCIUM
  //update at time (t + dt)
  if (microdomainNames.size() > 0)
  {
    updateMicrodomains_Ca();
    int numCpts = branchData->size;
    for (unsigned int ii = 0; ii < microdomainNames.size(); ii++)
    {//calculate RHS[] and Ca_microdomain[]
      int offset = ii * numCpts;
      Ca_microdomain_cur[offset] = Ca_microdomain[offset] = 2 * Ca_microdomain[offset] - Ca_microdomain_cur[offset];
    }
  }
#endif

#ifdef DEBUG_HH
	printDebugHH();
#endif
}

void CaConcentrationJunction::printDebugHH(std::string phase)
{
  std::cerr << "step,time|" << phase << " [rank,nodeIdx,instanceIdx] " <<
    "(neuronIdx,branchIdx,brchOrder){x,y,z,r | dist2soma,surfarea,volume,len} Vm" << std::endl;
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  std::cerr << getSimulation().getIteration() << "," 
            << getSimulation().getIteration() * *getSharedMembers().deltaT
            << "| " << phase
            << " [" << getSimulation().getRank() << "," << getNodeIndex() << ","
            << getIndex() << "] "
            << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
            << _segmentDescriptor.getBranchIndex(branchData->key) << ","
            << _segmentDescriptor.getBranchOrder(branchData->key) << ") {"
            << dimensions[0]->x << "," 
            << dimensions[0]->y << ","
            << dimensions[0]->z << "," 
            << dimensions[0]->r << " | " 
            << dimensions[0]->dist2soma << "," << dimensions[0]->surface_area << ","
            << dimensions[0]->volume << "," << dimensions[0]->length << "} " 
            << Ca_new[0] << std::endl;

  Array<DimensionStruct*>::iterator diter = dimensionInputs.begin();
  Array<dyn_var_t*>::iterator vend = CaConcentrationInputs.end();
  int c = -1;

  std::cerr << "JCT_INPUT_i " <<
    "(neuronIdx,branchIdx,brchOrder, brType, COMPUTEORDER){x,y,z,r | dist2soma,surfarea,volume,len} Vm" << std::endl;
  Array<dyn_var_t*>::iterator viter = CaConcentrationInputs.begin();
  for (viter = CaConcentrationInputs.begin(); viter != vend; ++viter, ++diter)
  {
    c++;
    std::cerr << " JCT_INPUT_" << c 
      << "(" << _segmentDescriptor.getNeuronIndex(branchData->key) << ","
      << std::setw(2) << _segmentDescriptor.getBranchIndex(branchData->key) << ","
      << _segmentDescriptor.getBranchOrder(branchData->key) << ","
      << _segmentDescriptor.getBranchType(branchDataInputs[c]->key) << ","
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
bool CaConcentrationJunction::checkSite(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset,
    CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset)
{
  assert(dimensions.size() == 1);
  DimensionStruct* dimension = dimensions[0];
  TissueSite& site = CG_inAttrPset->site;
  bool rval = (site.r == 0);
  if (!rval) rval = ((site.r * site.r) >= DISTANCE_SQUARED(site, *dimension));
  return rval;
}

bool CaConcentrationJunction::confirmUniqueDeltaT(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset,
    CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset)
{
  return (getSharedMembers().deltaT == 0);
}


CaConcentrationJunction::~CaConcentrationJunction() {}


#ifdef MICRODOMAIN_CALCIUM
void CaConcentrationJunction::createMicroDomainData(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset) 
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

    int numCpts = branchData->size;
    Ca_microdomain.increaseSizeTo(numMicrodomains* numCpts);
    Ca_microdomain_cur.increaseSizeTo(numMicrodomains* numCpts);
    RHS_microdomain.increaseSizeTo(numMicrodomains * numCpts);
    volume_microdomain.increaseSizeTo(numMicrodomains * numCpts);
    //currentDensityToConc_microdomain.increaseSizeTo(numMicrodomains * numCpts);
#if MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_NTSMACRO
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
#if MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_NTSMACRO
      //v_efflux[ii] = V_EFFLUX;
      //VOLUME_MICRODOMAIN
      if (tokens[ii] == "domain1")
      {
        v_efflux[ii] = V_EFFLUX_DOMAIN1; // [1/ms]
        //volume_microdomain[ii] = dimensions[0]->volume * VOLUME_MICRODOMAIN1;
        volume_microdomain[ii] = dimensions[0]->surface_area * FRACTION_SURFACEAREA_MICRODOMAIN1 * DEPTH_MICRODOMAIN1 * 1e-3;  // [um^3]
      }
      if (tokens[ii] == "domain2")
      {
        v_efflux[ii] = V_EFFLUX_DOMAIN2;
        //volume_microdomain[ii] = dimensions[0]->volume * VOLUME_MICRODOMAIN2;
        volume_microdomain[ii] = dimensions[0]->surface_area * FRACTION_SURFACEAREA_MICRODOMAIN2 * DEPTH_MICRODOMAIN2 * 1e-3;  // [um^3]
      }
      if (tokens[ii] == "domain3")
      {
        v_efflux[ii] = V_EFFLUX_DOMAIN3;
        //volume_microdomain[ii] = dimensions[0]->volume * VOLUME_MICRODOMAIN3;
        volume_microdomain[ii] = dimensions[0]->surface_area * FRACTION_SURFACEAREA_MICRODOMAIN3 * DEPTH_MICRODOMAIN3 * 1e-3;  // [um^3]
      }
#elif MICRODOMAIN_DATA_FROM == _MICRODOMAIN_DATA_FROM_CHANPARAM
      volume_microdomain[ii] = dimensions[0]->surface_area * domainData["fraction_surface"][0] * domainData["depth_microdomain"][0] * 1e-3;  // [um^3]
      v_efflux[ii] = domainData["v_efflux"][0];
#endif
    }
  }
}

void CaConcentrationJunction::setupCurrent2Microdomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset) 
{
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

void CaConcentrationJunction::setupFlux2Microdomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset) 
{
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

void CaConcentrationJunction::setupReceptorCurrent2Microdomain(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CaConcentrationJunctionInAttrPSet* CG_inAttrPset, CG_CaConcentrationJunctionOutAttrPSet* CG_outAttrPset) 
{
  //put channel producing Ca2+ influx to the right location
  //from that we can update the [Ca2+] in the associated microdomain
  CustomString microdomainName = CG_inAttrPset->domainName;
  int ii = 0;
  while (microdomainNames[ii] != microdomainName)
  {
    ii++;
  }
  _mapReceptorCurrentToMicrodomainIndex[receptorCaCurrents_microdomain.size()-1] = ii;
}

// GOAL: update RHS[], RHS_microdomain[], Ca_microdomain[] at time (t + dt/2)
void CaConcentrationJunction::updateMicrodomains(double& LHS, double& RHS)
{
  Array<ChannelCaCurrents>::iterator citer;
  Array<ChannelCaCurrents>::iterator cend ;
  int numCpts = branchData->size;//only 1 compartment in Junction
  unsigned int ii = 0;
  dyn_var_t bmt= getSharedMembers().x_bmt; // [1/ms]
  for (ii = 0; ii < microdomainNames.size(); ii++)
  {
    int offset = ii * numCpts;
    for (int jj = 0; jj < numCpts; jj++ )
    {
      RHS_microdomain[jj+offset] = ((bmt * volume_microdomain[ii] / dimensions[jj]->volume) *
         Ca_microdomain[jj+offset]) + v_efflux[ii] * Ca_new[jj];  // [uM/ms]
    }
  }
  citer = channelCaCurrents_microdomain.begin();
  cend = channelCaCurrents_microdomain.end();
  // loop through different kinds of Ca2+ currents via channels (LCCv12, LCCv13, R-type, ...)
  //  I_Ca [pA/um^2]
  ii = 0;
  for (; citer != cend; ++citer, ++ii)
  {
    int offset = _mapCurrentToMicrodomainIndex[ii] * numCpts;
    for (int jj = 0; jj < numCpts; jj++)
    {
      RHS_microdomain[offset+jj] -= currentDensityToConc * (*(citer->currents))[jj];  //[uM/ms]
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

  Array<dyn_var_t*>::iterator riter = receptorCaCurrents_microdomain.begin();
  Array<dyn_var_t*>::iterator rend = receptorCaCurrents_microdomain.end();
  // loop through different kinds of Ca2+ currents via receptors (NMDAR, ...)
  //  I_Ca [pA/um^2]
  ii = 0;
  for (; riter!= rend; ++riter, ++ii)
  {
    int offset = _mapReceptorCurrentToMicrodomainIndex[ii] * numCpts;
    for (int jj = 0; jj < numCpts; jj++)
    {
      RHS_microdomain[offset+jj] -= currentDensityToConc * (*(*riter));  //[uM/ms]
      //RHS_microdomain[offset+jj] -= currentDensityToConc * (*(*riter))[jj];  //[uM/ms]
    }
  }
  for (unsigned int ii = 0; ii < microdomainNames.size(); ii++)
  {//calculate RHS[] and Ca_microdomain[]
    int offset = ii * numCpts;
    //finally [NOTE: calculate Ca_microdomain can move to doBackwardSolve()]
    ////option1 to calculate Ca_microdomain
    //  Ca_microdomain[ii] = (RHS_microdomain[ii] + v_efflux[ii] * Ca_new[0]) 
    //    / (LHS + v_efflux[ii]);
    ////option2 to calculate Ca_microdomain
    ///Ca_microdomain[offset] = (RHS_microdomain[offset] - 
    ///    v_efflux[ii]/2.0 * (Ca_microdomain[offset]) + v_efflux[ii] * Ca_new[0]) /
    ///  (LHS + v_efflux[ii]/2.0);
    RHS +=  v_efflux[ii] * (Ca_microdomain[offset] ); //J_efflux(Cadomain--> Camyo)
    LHS += v_efflux[ii];
    //RHS_microdomain[offset] -=  v_efflux[ii] * (Ca_microdomain[offset] - Ca_new[0]); //J_efflux(Cadomain--> Camyo)
  }
}
void  CaConcentrationJunction::updateMicrodomains_Ca()
{ 
  dyn_var_t bmt = getSharedMembers().x_bmt; // [1/ms]
  int numCpts = branchData->size;
  for (unsigned int ii = 0; ii < microdomainNames.size(); ii++)
  {//calculate RHS[] and Ca_microdomain[]
    int offset = ii * numCpts;
    //finally [NOTE: calculate Ca_microdomain can move to doBackwardSolve()]
    ////option1 to calculate Ca_microdomain
    //  Ca_microdomain[ii] = (RHS_microdomain[ii] + v_efflux[ii] * Ca_new[0]) 
    //    / (LHS + v_efflux[ii]);
    ////option2 to calculate Ca_microdomain
    double LHS = bmt * volume_microdomain[ii] / dimensions[0]->volume + v_efflux[ii];
    //Ca_microdomain[offset] = (RHS_microdomain[offset] - 
    //    v_efflux[ii]/2.0 * (Ca_microdomain[offset]) + v_efflux[ii] * Ca_new[0]) /
    //  (LHS + v_efflux[ii]/2.0);
     Ca_microdomain[offset] = (RHS_microdomain[offset] / LHS) ;
  }
}
#endif
