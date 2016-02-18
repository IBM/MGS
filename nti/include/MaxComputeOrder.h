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
// Physical constants
//
#define zF  96485.3399  //[C/mol]=[mJ/(mV.mol)] - Faraday constant
#define zR  8.314472e3 //[mJ/(K.mol)] - universal constant
#define zCa 2          // valance of Ca2+ ions
#define zCa2F2_R ((zCa*zCa)*(zF*zF)/(zR))
#define zCaF_R (zCa*zF/(zR))

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

  // the fraction of cross-surface area of the volume occupied by cyto
  //               compared to that occupied by total compartment volume
#define FRACTION_CROSS_SECTIONALAREA_CYTO 0.5 
  
#define FRACTION_CROSS_SECTIONALAREA_ER 0.032 
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


///////////////////////////////////////////////////////////////////////
// list of paper models
#define _COMPONENT_UNDEFINED    0
//{{{ Na-models
//Na-transient
#define NAT_HODGKIN_HUXLEY_1952 1
#define NAT_WOLF_2005           2
#define NAT_HAY_2011            3
#define NAT_SCHWEIGHOFER_1999   4
// Na-persistent
#define NAP_WOLF_2005           2
//}}}
//{{{ K-models
// KAf
#define KAf_WOLF_2005          2
// KAs
#define KAs_WOLF_2005          2
// KIR
#define KIR_WOLF_2005          2
// KRP
#define KRP_WOLF_2005          2
// KDR
#define KDR_HODGKIN_HUXLEY_1952 1
#define KDR_SCHWEIGHOFER_1999   4

// BK-alpha
// BK-alphabeta
// NOTE: Those with the same values are indeed the same model
//        just being used in different papers
#define BKalphabeta_SHAO_1999       2       
#define BKalphabeta_WOLF_2005       2       
// SK
#define SK_MOCZYDLOWSKI_1993 2
#define SK_WOLF_2005    2
//}}}
//{{{ Ca-models
// CaN
#define CHANNEL_CaN _COMPONENT_UNDEFINED
// CaPQ
#define CHANNEL_CaPQ _COMPONENT_UNDEFINED
// CaR
#define CHANNEL_CaR _COMPONENT_UNDEFINED
// CaT
#define CaT_GHK_WOLF_2005 2
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
  #define CHANNEL_KAf KAf_WOLF_2005
  #define CHANNEL_KAs KAs_WOLF_2005
  #define CHANNEL_KIR KIR_WOLF_2005
  #define CHANNEL_KRP KRP_WOLF_2005
  #define CHANNEL_BKalphabeta  BKalphabeta_WOLF_2005
  #define CHANNEL_SK SK_WOLF_2005
  #define CHANNEL_CaT CaT_GHK_WOLF_2005
#else
  NOT IMPLEMENTED YET
#endif

//////////////////////////////////////////////////////////////////////
// Default setting
//
#ifndef CHANNEL_NAT
  #define CHANNEL_NAT _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_NAP
#define CHANNEL_NAP _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_KAf
#define CHANNEL_KAf _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_KAs
#define CHANNEL_KAs _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_KIR
#define CHANNEL_KIR _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_KRP
#define CHANNEL_KRP _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_BKalpha
#define CHANNEL_BKalpha _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_BKalphabeta
#define CHANNEL_BKalphabeta _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_SK
#define CHANNEL_SK _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_KDR
#define CHANNEL_KDR _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_CaT
#define CHANNEL_CaT _COMPONENT_UNDEFINED
#endif
#ifndef MODEL_TO_USE
#define MODEL_TO_USE _MODEL_NOT_DEFINED
#endif
#ifndef SIMULATION_INVOLVE
#define SIMULATION_INVOLVE  VMONLY
#endif
#endif //_MAXCOMPUTEORDER_H
