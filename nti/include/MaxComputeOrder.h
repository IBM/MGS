// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
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
//{{{
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

//}}}

//NOTE: disable this if we want 'distance' information is kept in Touch
//  LTWT = light-weight    (NOTE: there is still bug with heavy-weight)
#define LTWT_TOUCH

//Debug purpose
//{{{
//  DEBUG_COMPARTMENT (if define, it helps to analyze when a cpt variable becomes NaN)
//#define DEBUG_COMPARTMENT
//
//#define DEBUG_HH
//// if enabled, print Hodgkin-Huxley information

// if enabled, it always check for NaN or values outside the expected range
#ifndef DEBUG_ASSERT
#define DEBUG_ASSERT
#endif

//  IDEA_DYNAMIC_INITIALVOLTAGE (if defined, it enables us to pass different voltage value at different location of branch tree - this helps to reach equilibrium faster on neuron where gradient voltage occurs)
//#define IDEA_DYNAMIC_INITIALVOLTAGE  // can be defined inside NTSMacros.h within the MODEL_TO_USE section

//#define DEBUG_CPTS  //this option should be set via gsl/configure.py
//   It tells to generate code that print out 
//   statistical data about mean +/- SD of
//     1. surfaceArea
//     2. volume
//     3. length
//  in the following branch-types
//     A. AXON
//     B. BASALDEN
//     C. APICALDEN
     
//#define TD_DEBUG  //put inside TouchDetector.cxx - to debug find what Segments has multiple touches with Spines(head/neck)
//}}}

//NOTE: 
///enable or disable the macro to turn on or off the option
//  IDEA1 = the rescale of explicit junction compartment by taking capsules from 
//  NEWIDEA = (Params.cxx) which is mainly to deal with accept param's value as a string, which can be the name of the function with parameters for that function, e.g.
//  gbar = lineardistane(a,b) 
//  gbar = linearbranchorder
//  ...
//  TOUCHDETECT_SINGLENEURON_SPINES (if defined, it will use the strategy developed to ensure proper binding to the right location on the branch compartment) - DO NOT defined this when using full tissue 
//#define RESAMPLING_SPACE_VOLUME (if defined, it resample based on a given space distance and volume for a given tolerance, i.e. (dist-space) < dist_tolerance, (volume-volCrit) < volTolerance)
//#define IDEA1
//#define NEWIDEA
//#define SYNAPSE_PARAMS_TOUCH_DETECT
//#define INFERIOR_OLIVE
//#define TOUCHDETECT_SINGLENEURON_SPINES
//#define RESAMPLING_SPACE_VOLUME
//#define SPINE_HEAD_UNIQUE_TOUCH // put inside TouchDetector.cxx - to ensure 1 spine head get 1 synaptic cleft

//different strategy when modeling soma
//{{{
// STRATEGY 01 - simply treate as a single point (i.e. no volume is considered)
//  use either (but NOT both) USE_SOMA_AS_ISOPOTENTIAL or USE_SOMA_AS_POINT
#define USE_SOMA_AS_ISOPOTENTIAL //enable this when we want to simulate the soma as a well-mixed iso-potential compartment - so that the distance from the first-proximal compartment to soma is actually the distance from center node to outer surface of the spherical soma, i.e. half-length of the dendritic/axonic compartment

//#define USE_SOMA_AS_POINT   //enable this when we want to simulate the soma as 1um^2 surface-area regardless of area being used in the morphology

// STRATEGY 02 [experimental] - adjust the soma to consider 'effect' in that the real-shape soma may obstruct the
//               propagation of electrical signal than the spherical one
//#define USE_STRETCH_SOMA_RADIUS   //which use STRETCH_SOMA_RADIUS_WITH
#ifndef STRETCH_SOMA_WITH   // only work when disable STRATEGY 01 
#define STRETCH_SOMA_WITH 0.0    
//#define STRETCH_SOMA_WITH 40.0 // [um] - make soma longer (hoping to make diffusion slower)
//#define STRETCH_SOMA_WITH 50.0 
//#define STRETCH_SOMA_WITH 0.0 
//#define STRETCH_SOMA_WITH 130.0 
//#define STRETCH_SOMA_WITH 50.0 
//#define STRETCH_SOMA_WITH 25.0    
#endif

// STRATEGY 03 [experimental]- make the neck smaller is another way to treat if real-shape soma obstruct the 
//               propagation of electrical signal than the spherical one
//{{{
//#define USE_SCALING_NECK_FROM_SOMA
#ifndef SCALING_NECK_FROM_SOMA_WITH
#define SCALING_NECK_FROM_SOMA_WITH 1.0  //>1: make neck smaller
#endif
//}}}

// STRATEGY for Calcium in Soma calculation
//{{{ decide if we want to simulate Ca2+ in soma as the concentration in sub-shell volume or not
//  here it use 2 parameters
//      1. SHELL_DEPTH - the depth of the cell from the bio-membrane
//      2. THRESHOLD_SIZE_R_SOMA - the minimal size to consider soma-Neuron 
//                       to distinguish from 'soma'-spineHead
//#define USE_SUBSHELL_FOR_SOMA
#ifdef USE_SUBSHELL_FOR_SOMA
#define SHELL_DEPTH  1.0  // [um]
#define THRESHOLD_SIZE_R_SOMA  2.0 // [um]
#endif
//}}}


// Add internal noise to each neuron (junction only, i.e. soma)
//  currently the parameter are hard-coded (we may want to pass via HodgkinHuxleyVoltageJunction)
// #define INTRINSIC_NOISE_TO_NEURON

// if enabled, we can define ArrayCurrentPulseGenerator 
// and connect to HodgkinHuxleyVoltageJunction (i.e. soma only)
// by doing this, we just create 1 pulse-generator that is capable of 
//  generating different streams of currents to create heterogeneity
// #define CURRENT_STREAMS_TO_NEURON
//}}}

//numerics
//{{{numerics
// STRATEGY for 'junction' compartment
//#define NEW_RADIUS_CALCULATION_JUNCTION    //if defined; then at junction Rb=(*diter)->r
                        // if not; then Rb = ((*diter)->r + dimension->r)/2
                        
//#define CONSIDER_EFFECT_LARGE_CHANGE_CURRENT_STIMULATE
//  IN a real-world system, the current injected is 'sensed' by the soma first;
//  before any significant propagation
//  In NEURON, the soma is modeled as a cylinder, and thus it has 2 'end' nodes only
//  In NTS, the soma is modeled as a sphere, and thus it can has many nodes 
//     which makes the injected current less available to the soma center point
//  Because of that, we can design a system to consider the injected current
//     available to a smaller region, rather than to the whole-soma
//     due to the cytosolic resistance 



//{{{ terminal point
//#define USE_TERMINALPOINTS_IN_DIFFUSION_ESTIMATION //if defined, then 
         // (suppose Vm[size-1]) instead of using proximalVoltage, and the distance between them
         // it use V0 (or Vterminal_proximal) and distance as length/2
         // and Vterminal_proximal is estimated using algebraic equation
         // Vterminal_proximal= (w1 * proximalVm + w2 * Vm[size-1])
         //  with weight is inverse of distance
         //     w1 = 1/(proximalDimension->length)
         //     w2 = 1/(dimensions[size-1]->length)
//}}}

//{{{ switch between Kozloski-2011 implementation of discritization vs. original paper's method
// The Mascagni (1995) is supposed to be the right version
#define NEW_DISTANCE_NONUNIFORM_GRID //if defined, then ensure 
//       dsi = (dx1 + dx2)/2 - Check Mascagni (1995, pg 33)
//}}}

// When calculating ionic current that is non-linear of voltage Vm
//   then I(t+dt/2)  ~  I(t) + di/dv * (V(t+dt/2) - V(t))
// This is more accurate than just using I(t)
#define CONSIDER_DI_DV      
// use  series resistance Rs in Voltage-clamp (i.e. ideal single-electrode clamp) 
//      rather than (double-electrode clamp) using headstage-gain beta and specific membrane capacitance Cm
#define USE_SERIES_RESISTANCE
//}}}

//point-process
//{{{
// as a point process - the injected current directly affect the center point, without distributing to all 
//   locations on the compartment, i.e. no need to divide by the surface area
// This is critical 
//#define INJECTED_CURRENT_IS_POINT_PROCESS 
//}}}

//Spine options
//{{{Spine options

#define SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM //if defined, then the user can specify what compartments is neck or head of the spine via SynParams.par in  COMPARTMENT_SPINE_NECK, COMPARTMENT_SPINE_HEAD
       
//choices for how data is exchanged when we couple spines to shaft
//{{{ 
//BY DEFAULT: 
//  the current between spine-neck and den-shaft is modeled as injectedCurrent producer
//    Iinj(to-neck, time=t+dt/2) = g * (Vshaft(t) - Vneck(t)) 
//  which is explicit method - and may crash when there are many spines onto a single compartment
// Consider these different improvement below:

//#define CONSIDER_MANYSPINE_EFFECT_OPTION1 
//  // if defined, the new codes handle the case 
//   when there are many spines conntact to one compartment; and thus the amount of Vm or Ca2+ 
//   propagate to the nneck needs to be equally divided  (this is important for numerical stability)
//   HOWEVER: The method is explicit; and thus is not accurate as well
       // NOTE: DO NOT use both with CONSIDER_MANYSPINE_EFFECT_OPTION2

#define CONSIDER_MANYSPINE_EFFECT_OPTION2 //option 2 means we convert into 
// ConductanceProducer and ReversalPotential producer
//    Ichannel(to-neck, time=t+dt/2) = g_density * (Vneck(t+dt/2) - Vshaft(t)) 
//    Ichannel(to-shaft, time=t+dt/2) = g_density * (Vshaft(t+dt/2) - Vneck(t)) 
// HOWEVER: the calculation may not be accurate due to the fact that 
       // NOTE: DO NOT use both with CONSIDER_MANYSPINE_EFFECT_OPTION1
#define CONSIDER_MANYSPINE_EFFECT_OPTION2_CACYTO 
#define CONSIDER_MANYSPINE_EFFECT_OPTION2_CAER
     //When many spines connect to a single compartment

//#define CONSIDER_MANYSPINE_EFFECT_OPTION2_revised //option2 revised
//  is the combination of OPTION2 and OPTION1
//   1. the shaft's signal's loss is split across the many spines connecting to it
//    Ichannel(to-neck, time=t+dt/2) = g_density_new * (Vneck(t+dt/2) - Vshaft(t)) 
//   with g_density_new = g_density / num_spine_connecting_to_shaft
//   --> 'rate' of signal loss is reduced
//   NOTE: Require defining of OPTION2; and disable OPTION1

//#define CONSIDER_MANYSPINE_EFFECT_OPTION2_PREDICTOR_CORRECTOR 
//  This need to combine with OPTION2 above
//  Basically, for each compartment that has many spines
//       it calculate Vnew_shaft[t+dt/2] using  Vspines[t]
//       it calculate Vspine[t+dt/2] using  Vcur_shaft[t]
//  Corrector:  
//    Ichannel(to-neck, time=t+dt/2) = g_density * (Vneck(t+dt/2) - 1/2 * (Vshaft(t) + V^*shaft(t+dt/2))) 
//    Ichannel(to-shaft, time=t+dt/2) = g_density * (Vshaft(t+dt/2) - 1/2 * (Vneck(t) + V^*neck(t+dt/2)) 
//}}}
#define KEEP_PAIR_PRE_POST   // this ensure a receptor always produce the pre- and post- branch information via only 1 interface; 
 // ..  rather than 2 separate set of interface 
 // .. ALWAYS ENABLE THIS: as we haven't made produce post-index available yet 
 // NOTE: This is 2-element array: with pre-side first then post-side
// The reason to have this is to enable AnyConcentrationDisplay to be able to 
//   capture [NT] of a particular type of Neurotransmitter (e.g. Glut, GABA)
// We also needs to update Synapse-receptors's Interfaces

//#define SUPPORT_MODULABLE_CLEFT  //enable this if we want to have DA, Ser(otonin) as part of neurotransmitter in the SynapticCleft Node

//NOTE: This is for the choice of defining input to/output from SynapticReceptor
//      inside TissueFunctor
//  [NMDAR] [Voltage] [Voltage, Calcium]
// case 1 (RECEPTOR_PRE_AS_INPUT_POST_AS_INPUT_OUTPUT)
//   mean [Voltage(pre, input)] [Voltage(post,input+output), Calcium(post,input+output)]
//#define RECEPTOR_PRE_AS_INPUT_POST_AS_INPUT_OUTPUT
// case 2 (otherwise)
//   mean [Voltage(post, input)] [Voltage(post,output), Calcium(post,output)]
//#else //RECEPTOR_POST_AS_INPUT_POST_AS_OUTPUT
//     In this case, the SynapticCleft is hardcoded to receive 'Voltage'
//     which is used for calculating [NT]

//by default, a single capsule can be used to create a single SynapticCleft maximum. This is however, a limitation when we build a network of single-compartmental neuron
//where we want a neuron can form multiple SynapticClefts with other neurons through its single capsule
//#define SINGLE_JUNCTIONAL_CAPSULE_CAN_FORM_MULTIPLE_SYNAPSE   //tell TissueFunctor to enable multiple SynapticCleft instances from a single junctional pre-capsule
//TODO: may need to update for compartmental capsule? (currently not having a user-case)
//}}}

//IP3 modeling
//{{{ modeling IP3 concentration
#define IP3_DIFFUSIONAL_VAR 3
// rather than modeling IP3 as a diffusional variable; we can 
// ...just consider it is a scalar variable
// ...and then either put IP3 into 
#define IP3_INSIDE_IP3R 1
#define IP3_INSIDE_CLEFT 2
//CHOICE
#define IP3_LOCATION IP3_DIFFUSIONAL_VAR

#if IP3_LOCATION == IP3_INSIDE_IP3R
// [IP3] is part of the channels
//            to avoid modeling IP3 production, IP3 as a compartmental variable
//     As IP3 production is a function of mGluR activation
//       as we already have [Glut] - in the cleft - whose dynamics affect mGluR activation
#ifdef IP3_MODELAS_FUNCTION_GLUT
//     We model [IP3] ~ f([Glut])
#endif
//#define IP3_MODELAS_FUNCTION_GLUT // this is used in ChannelIP3 model which avoid having IP3 as explicit compartmental variable & the [IP3] is a function of [Glut] in the synapse
//    ALWAYS DISABLE this for now
#endif
//}}}

//Microdomain calcium
//{{{ MICRODOMAIN_CALCIUM
//#define MICRODOMAIN_CALCIUM  //if defined, then the system enable the capability to model microdomain calcium volume where ion can flow into this first before going to the cytosolic bath --> maybe we can use this to avoid the sub-shell feature
// STATUS:
//     The compartment needs to know how many 'microdomains' it contains and create them before any connection to/from channels
//  LIMITATION:   --> it uses information from 'ChanParams.par' only
//       ISSUE: if receptors can form the microdomain between them, i.e. no channel connect to such microdomain, then it fails
//       PLAN-9.2.178.66:  add the capability --> read microdomain also from Synparams.par
// STATUS: Only accept 'Channels' to connect to 'Compartments[Calcium]' or 'Junctions[Calcium]'
//         Channels can produce HH-current or GHK-current 
//  ADDED:   --> accept Receptor connecting to microdomain
//       DONE: allow receptor form microdomain with channels
//       PLAN: allow receptor form microdomain with receptors (this require PLAN-9.2.178.66) 
// Now we need to decide where to pass in data for v_efflux and volume_microdomain
#define _MICRODOMAIN_DATA_FROM_NTSMACRO   0 
#define _MICRODOMAIN_DATA_FROM_CHANPARAM  1
#define MICRODOMAIN_DATA_FROM _MICRODOMAIN_DATA_FROM_CHANPARAM
// This needs to be implemented in ChanParam if _MICRODOMAIN_DATA_FROM_CHANPARAM is used
//#MICRODOMAIN_PARAMS 2
//## v_efflux in [1/ms]
//## volume_microdomain in [% of cytosolic-Ca2+]
//#domain 1
//#BRANCHTYPE
//#[1:2]  <v_efflux=1.4; volume_microdomain=0.2>
//#domain2 1
//#BRANCHTYPE
//#[1:2]  <v_efflux=1.4; volume_microdomain=0.02>

//NOTE: Use this if _MICRODOMAIN_DATA_FROM_NTSMACRO is used
//#define GENERATE_V_EFFLUX(Argument V_EFFLUX_##Argument
//default value if MICRODOMAIN_DATA_FROM_NTSMACRO is used
//#define V_EFFLUX  0.1    //[1/ms]
//Here, we assume maximum 3 microdomains on 1 branch/junction
//               and the microdomains must be named using 'domain1', 'domain2', 'domain3'
//NOTE: v_efflux in Ca-subspace use 0.25 [1/ms]
// the rate also reflect the binding affinity (indirectly)
#define V_EFFLUX_DOMAIN1  0.1    //[1/ms]
#define V_EFFLUX_DOMAIN2  0.1    //[1/ms]
#define V_EFFLUX_DOMAIN3  0.1    //[1/ms]
//#define V_EFFLUX_DOMAIN1  0.00    //[1/ms] - make small to emulate sustain Ca2+ binding to KChIP (should move to parameter file)
//#define V_EFFLUX_DOMAIN2  0.00    //[1/ms]
//#define V_EFFLUX_DOMAIN3  0.00    //[1/ms]
#define DEPTH_MICRODOMAIN1  10.0  //[nanometer]
#define DEPTH_MICRODOMAIN2  10.0  //[nanometer]
#define DEPTH_MICRODOMAIN3  10.0  //[nanometer]
#define FRACTION_SURFACEAREA_MICRODOMAIN1 1.0  //[% of membrane surface area]
#define FRACTION_SURFACEAREA_MICRODOMAIN2 1.0  //[% of membrane surface area]
#define FRACTION_SURFACEAREA_MICRODOMAIN3 1.0  //[% of membrane surface area]
//}}}

//Global parameters
// //all compartments have the same Vhalf_(m/h)_adjust
//#define TURN_ON_ADJUST_VHALF_SIMPLE
// //each compartment may have the different Vhalf_(m/h)_adjust
// (not implemented)
//#define TURN_ON_ADJUST_VHALF_COMPARTMENT
// // pass tau(Vm) as explicit parameters
// // then each branch-type may have different tau(Vm)
// // but all compartments in that branchtype receive same tau(Vm)
//#define TURN_ON_ADJUST_TIMECONSTANT
// // if defined, then the parameters (e.g. those in the ODE/PDE)
// // are saved when an associated trigger is activated
// // and then restore when an associated trigger is activated
//#define ENABLE_STORE_AND_RESET_PARAMETERS

//I/O options
//{{{I/O options
//NOTE: this is for adaptive I/O using sensor
#define _SINGLE_SENSOR_DETECT_CHANGE  1
#define _MULTIPLE_SENSORS_DETECT_CHANGE  2
#define DETECT_CHANGE _MULTIPLE_SENSORS_DETECT_CHANGE

//#define WRITE_GATES  --> if enable globally or within each ChannelXXX model
//   we can trigger the writing out of the gates (m, h, n, ...) for model using
//             Hodgkin-Huxley-based formula
//   NOTE: currently supported in ChannelKAs

//  IDEA_ILEAK (if defined, the code that enable outputing Ileak is added to AnyCurrentDisplay via HodgkinHuxleyVoltage connection)
//  IDEA_CURRENTONCOMPT (if defined, we can output the current on any compartments on any branch by providing the 'site')
//#define IDEA_ILEAK
//#define IDEA_CURRENTONCOMPT

// RECORD_AXIAL_CURRENT_AS_INJECTED_CURRENT enable this if we want to track the total axial current 
//    flowing into soma as (pA)
//    = Area(soma) * SUM( {Vdend - Vsoma}/ (r_dend_soma) )
//#define RECORD_AXIAL_CURRENT_AS_INJECTED_CURRENT
//}}}

// Other extensions
//{{{
//#define NVU_NTS_EXTENSION  //we need this to enable TissueFunctor to work with NVU
//#define DYNAMIC_ECS_Sodium
//#define DYNAMIC_ECS_Potassium
//#define DYNAMIC_ECS_Temperature
////with this, the ExtracellularMedium is nolonger be used, instead
//                     //we use ECSNode
//}}}
#endif //_MAXCOMPUTEORDER_H
