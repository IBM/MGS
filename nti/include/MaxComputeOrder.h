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
// Numerical settings
//{{{
#define MAX_COMPUTE_ORDER 7 
#define USE_DOUBLES 
//g++ -DUSE_DOUBLES=1 myprogram.cpp
//dyn_var_t = dynamical-variable-type
#ifdef USE_DOUBLES
//typedef double dyn_var_t;
 #define dyn_var_t double
#else
//typedef float dyn_var_t;
 #define dyn_var_t float
#endif
//typedef double key_size_t
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

//NOTE: distable this if we want 'distance' information is kept in Touch
//  LTWT = light-weight
#define LTWT_TOUCH

//#define SYNAPSE_PARAMS_TOUCH_DETECT
//#define INFERIOR_OLIVE

#endif //_MAXCOMPUTEORDER_H
