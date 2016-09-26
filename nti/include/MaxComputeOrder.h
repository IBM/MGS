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
// ================================================================
#ifndef _MAXCOMPUTEORDER_H
#define _MAXCOMPUTEORDER_H

#include "NTSMacros.h"
/*
#include <climits>
#include <cstdint>
#include <sys/types.h>
*/

///////////////////////////////////////////////////////////////////////
// NOTE: use WAIT_FOR_REST to ensure no current is activated
//       during the time the Vm is resetle to proper resting value
//       This should be applied to Ca2+ channels
//#define WAIT_FOR_REST 
//#ifdef WAIT_FOR_REST
//#define NOGATING_TIME 20.0 //ms
//#else
//#define NOGATING_TIME 0.0 //ms
//#endif
#ifndef NOGATING_TIME
#define NOGATING_TIME 0.0 //ms
#endif
// Numerical settings
//{{{
#ifndef MAX_COMPUTE_ORDER
#define MAX_COMPUTE_ORDER 0 
#endif
//#define USE_DOUBLES 

//NOTE: g++ -DUSE_DOUBLES=1 myprogram.cpp
//NOTE: dyn_var_t = dynamical-variable-type
//NOTE: we can't use typedef as MDL currently does not support type with new name like 'dyn_var_t'
//   e.g. typedef double dyn_var_t;
//   e.g. typedef float dyn_var_t;
//   e.g. typedef double key_size_t
#ifdef USE_DOUBLES
 #define dyn_var_t double
#else
 #define dyn_var_t float
#endif
//NOTE: both must match the size: double == unsigned long long
#define key_size_t double
#define key_mask_size_t unsigned long long
//}}}

// For Markov processing purpose
#define BITSHIFT_MARKOV 16 // bits ~ 2bytes ~ short type
#define MASK_MARKOV 0xffff
//typedef int16_t MarkovState_t ;
//typedef int32_t Combined2MarkovState_t; // 'long' for now
#define MAXRANGE_MARKOVSTATE  SHRT_MAX
//#define MarkovState_t short // few states
#define ClusterStateIndex_t int // or long??

//x = row, y=col
//WIDTH=#col, HEIGHT=#row
#ifndef Map1Dindex
#define Map1Dindex(x,y, WIDTH) ((y)+(x)*(WIDTH))
#endif

#ifndef Find2Dindex
#define Find2Dindex(x,y, i, WIDTH) \
		do{\
			(y) = (i) % (WIDTH);\
			(x) = (i) / (WIDTH);\
		}while (0)
#endif

//NOTE: disable this if we want 'distance' information is kept in Touch
//  LTWT = light-weight    (NOTE: there is still bug with heavy-weight)
#define LTWT_TOUCH

//NOTE: 
///enable or disable the macro to turn on or off the option
//  IDEA1 = the rescale of explicit junction compartment by taking capsules from 
//  NEWIDEA = (Params.cxx) which is mainly to deal with accept param's value as a string, which can be the name of the function with parameters for that function, e.g.
//  gbar = lineardistane(a,b) 
//  gbar = linearbranchorder
//  ...
//  IDEA_ILEAK (if defined, the code that enable outputing Ileak is added to AnyCurrentDisplay via HodgkinHuxleyVoltage connection)
//  IDEA_CURRENTONCOMPT (if defined, we can output the current on any compartments on any branch by providing the 'site')
//  DEBUG_COMPARTMENT (if define, it helps to analyze when a cpt variable becomes NaN)
//  IDEA_DYNAMIC_INITIALVOLTAGE (if defined, it enables us to pass different voltage value at different location of branch tree - this helps to reach equilibrium faster on neuron where gradient voltage occurs)
//  TOUCHDETECT_SINGLENEURON_SPINES (if defined, it will use the strategy developed to ensure proper binding to the right location on the branch compartment) - DO NOT defined this when using full tissue 
//#define RESAMPLING_SPACE_VOLUME (if defined, it resample based on a given space distance and volume for a given tolerance, i.e. (dist-space) < dist_tolerance, (volume-volCrit) < volTolerance)
//#define IDEA1
//#define NEWIDEA
//#define IDEA_ILEAK
//#define IDEA_CURRENTONCOMPT
//#define SYNAPSE_PARAMS_TOUCH_DETECT
//#define INFERIOR_OLIVE
//#define DEBUG_COMPARTMENT
//#define IDEA_DYNAMIC_INITIALVOLTAGE  // can be defined inside NTSMacros.h within the MODEL_TO_USE section
//#define TOUCHDETECT_SINGLENEURON_SPINES
//#define RESAMPLING_SPACE_VOLUME
//#define USE_SOMA_AS_POINT   //enable this when we want to simulate the soma as a single point
#ifndef STRETCH_SOMA_WITH   // only work when disable USE_SOMA_AS_POINT
#define STRETCH_SOMA_WITH 35.0    //seem ok  value if used
//#define STRETCH_SOMA_WITH 050.0 
//#define STRETCH_SOMA_WITH 40.0 // [um] - make soma longer (hoping to make diffusion slower)
//#define STRETCH_SOMA_WITH 0.0 
//#define STRETCH_SOMA_WITH 130.0 
//#define STRETCH_SOMA_WITH 50.0 
//#define STRETCH_SOMA_WITH 25.0    
#endif
#ifndef SCALING_NECK_FROM_SOMA
#define SCALING_NECK_FROM_SOMA 1.0  //>1: make neck smaller
//#define SCALING_NECK_FROM_SOMA 10.0  //>1: make neck smaller
#endif
#define NEW_RADIUS_CALCULATION_JUNCTION    //if defined; then at junction Rb=(*diter)->r
                        // if not; then Rb = ((*diter)->r + dimension->r)/2
//#define USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION //if defined, then 
         // (suppose Vm[size-1]) instead of using proximalVoltage, and the distance between them
         // it use V0 (or Vterminal_proximal) and distance as length/2
         // and Vterminal_proximal is estiamted using algebraic equation
         // Vterminal_proximal= (w1 * proximalVm + w2 * Vm[size-1])
         //  with weight is inverse of distance
         //     w1 = 1/(proximalDimension->length)
         //     w2 = 1/(dimensions[size-1]->length)

#define NEW_DISTANCE_NONUNIFORM_GRID //if defined, then ensure 
//       dsi = (dx1 + dx2)/2 - Check Mascagni (1995, pg 33)
//#define CONSIDER_MANYSPINE_EFFECT_OPTION1 // if defined, the new codes that handle the case when there are many spines conntact to one compartment; and thus the amount of Vm or Ca2+ propagate to the nneck needs to be equally divided  (this is important for numerical stability)
       // NOTE: DO NOT use both with CONSIDER_MANYSPINE_EFFECT_OPTION2

#define CONSIDER_MANYSPINE_EFFECT_OPTION2 //option 2 means we convert into 
       // ConductanceProducer and ReversalPotential producer
       // (instead of injectedCurrent producer)
       // NOTE: DO NOT use both with CONSIDER_MANYSPINE_EFFECT_OPTION1
#define CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO 
#define CONSIDER_MANYSPINE_EFFECT_OPTION2_CAER

#define IP3_MODELAS_FUNCTION_GLUT // this is used in ChannelIP3 model which avoid having IP3 as explicit compartmental variable & the [IP3] is a function of [Glut] in the synapse

// rather than modeling IP3 as a diffusional variable; we just consider it is a scalar variable
// However, the question is IP3 reside: 
#define IP3_INSIDE_IP3R 1
#define IP3_INSIDE_CLEFT 2
#define IP3_LOCATION IP3_INSIDE_IP3R

#define SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM //if defined, then the user can specify what compartments is neck or head of the spine via SynParams.par in  COMPARTMENT_SPINE_NECK, COMPARTMENT_SPINE_HEAD
       
#endif //_MAXCOMPUTEORDER_H
