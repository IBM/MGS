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
//#define IDEA1
//#define NEWIDEA
//#define IDEA_ILEAK
//#define IDEA_CURRENTONCOMPT
//#define SYNAPSE_PARAMS_TOUCH_DETECT
//#define INFERIOR_OLIVE
//#define DEBUG_COMPARTMENT


#endif //_MAXCOMPUTEORDER_H
