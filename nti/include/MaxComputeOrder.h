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

///////////////////////////////////////////////////////////////////////
/// General definition
#define _YES 1
#define _NO 0

///////////////////////////////////////////////////////////////////////
/// Define what models are available here
//{{{
#define _MODEL_NOT_DEFINED 0 
#define _WOLF_2005_MSN  1 
//}}}
// define 
#define MODEL_TO_USE _MODEL_NOT_DEFINED

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

///////////////////////////////////////////////////////////////////////
// Geometrical settings
// NOTE: You are not suppose to modified it
//   If you really want to, modify in 'MODEL DESIGN' section
//{{{
  // USAGE: compare the distance between 2 points, if it is smaller than this then
  //        treated as overlapped
  // unit: micrometer
#define DISTANCE_AS_OVERLAPPED 1.0E-6
  // USAGE: fraction of cytoplasmic volume vs total compartment volume
#define FRACTIONVOLUME_CYTO    0.5
  // USAGE: fraction of smooth endoplasmic reticulum volume vs total compartment volume
  //        (dendrite)
#define FRACTIONVOLUME_SmoothER    0.032
  // USAGE: fraction of rough endoplasmic reticulum volume vs total compartment volume
  //        (only soma)
#define FRACTIONVOLUME_RoughER    0.15
 
  // assume that the surface area of cytoplasm is the same as that of biomembrane
#define FRACTION_SURFACEAREA_CYTO 1.0 

  // the surface area of ER must be smaller than that of biomembrane
#define FRACTION_SURFACEAREA_SmoothER 0.5 
#define FRACTION_SURFACEAREA_RoughER 0.5 
//}}}

///////////////////////////////////////////////////////////////////////
//NOTE: Assign to one of this for further computation
//   The value order are important here
//{{{
#define VMONLY 1
#define VM_CACYTO 2
#define VM_CACYTO_CAER 3
#define VM_CACYTO_CAER_DOPA 4
//}}}
 //default
#define SIMULATION_INVOLVE  VMONLY


///////////////////////////////////////////////////////////////////////
// list of paper models
#define _COMPONENT_UNDEFINED    0
//{{{ Na-models
//Na-transient
#define NAT_HODGKIN_HUXLEY_1952 1
#define NAT_WOLF_2005           2
#define NAT_HAY_2011            3
#define NAT_SCHWEIGHOFER_1999   4
#define CHANNEL_NAT _COMPONENT_UNDEFINED
// Na-persistent
#define NAP_WOLF_2005           2
#define CHANNEL_NAP _COMPONENT_UNDEFINED
//}}}
//{{{ K-models
// KAf
#define CHANNEL_KAf _COMPONENT_UNDEFINED
// KAs
#define CHANNEL_KAs _COMPONENT_UNDEFINED
// KRP
#define CHANNEL_KRP _COMPONENT_UNDEFINED
// KDR
#define CHANNEL_KDR _COMPONENT_UNDEFINED
// BK
#define CHANNEL_BK _COMPONENT_UNDEFINED
// SK
#define CHANNEL_SK _COMPONENT_UNDEFINED
//}}}
//{{{ Ca-models
// CaN
#define CHANNEL_CaN _COMPONENT_UNDEFINED
// CaPQ
#define CHANNEL_CaPQ _COMPONENT_UNDEFINED
// CaR
#define CHANNEL_CaR _COMPONENT_UNDEFINED
// CaT
#define CHANNEL_CaT _COMPONENT_UNDEFINED
// CaLv12
#define CHANNEL_CaLv12 _COMPONENT_UNDEFINED
// CaLv13
#define CHANNEL_CaLv13 _COMPONENT_UNDEFINED
//}}}
// NMDAR???
#define CHANNEL_NMDAR _COMPONENT_UNDEFINED
// AMPAR???
#define CHANNEL_AMPAR _COMPONENT_UNDEFINED
// PMCA???
#define CHANNEL_PMCA _COMPONENT_UNDEFINED
// NCX???
#define CHANNEL_NCX _COMPONENT_UNDEFINED
// RYR???
#define CHANNEL_RYR _COMPONENT_UNDEFINED
// IP3R???
#define CHANNEL_IP3R _COMPONENT_UNDEFINED

///////////////////////////////////////////////////////////////////////
// MODEL DESIGN
// 1. to choose a model: select the proper value for MODEL_TO_USE
#define MODEL_TO_USE _WOLF_2005_MSN
// 2. select what compartmental variables to use
#define SIMULATION_INVOLVE VMONLY
// 3. to disable any channel from the model, just comment it out
#if MODEL_TO_USE == _WOLF_2005_MSN
  #define CHANNEL_NAT NAT_WOLF_2005
  #define CHANNEL_NAP NAP_WOLF_2005
#else
  NOT IMPLEMENTED YET
#endif

#endif //_MAXCOMPUTEORDER_H
