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

///////////////////////////////////////////////////////////////////////
// Numerical settings
//{{{
#define MAX_COMPUTE_ORDER 0 
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

#endif //_MAXCOMPUTEORDER_H
