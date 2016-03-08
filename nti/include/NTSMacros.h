#ifndef _NTSMacros_H
#define _NTSMacros_H


/// IMPORTANT: jump to USER-SELECTED section 

///////////////////////////////////////////////////////////////////////
/// General definition
#define _YES 1
#define _NO 0

///////////////////////////////////////////////////////////////////////
// Physical constants
//
//{{{
#define e0  1.602e-19   // Coulomb = [C] - the elementary charge for 1 univalent ion
#define zF  96485.3399  //[C/mol]=[mJ/(mV.mol)] - Faraday constant - total charges for 1 mole 
                        // of univalent ion
#define zR  8.314472e3 //[mJ/(K.mol)] - universal constant
#define zCa 2          // valance of Ca2+ ions
#define zNa 1          // valence of Na+ ions
#define zK  1          // valence of K+ ions
#define zkB  1.381e-23  // [J/K] = Joule/Kelvin = Boltzmann constant (R/N_A)
#define zN_A 6.022e23   // [1/mol] = number of molecuels/atoms/ions per mole - Avogadro number

#define zCa2F2_R ((zCa*zCa)*(zF*zF)/(zR))
#define zCaF_R (zCa*zF/(zR))
#define zF_RT (zF / (zR * *getSharedMembers().T))
#define mM2uM 1e3   // conversion factor
#define uM2mM 1e-3  // conversion factor

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

  // the surface area of ER may be a few times more than that of biomembrane
	// due to the folding structure
	// Liver hepatocyte
#define FRACTION_SURFACEAREA_SmoothER 8 
#define FRACTION_SURFACEAREA_RoughER 27
  // Pancreatic exocrine cell
//#define FRACTION_SURFACEAREA_SmoothER 8 
//#define FRACTION_SURFACEAREA_RoughER 27

  // the fraction of cross-surface area of the volume occupied by cyto
  //               compared to that occupied by total compartment volume
#define FRACTION_CROSS_SECTIONALAREA_CYTO 0.5 
  
#define FRACTION_CROSS_SECTIONALAREA_ER 0.032 
//}}}

///////////////////////////////////////////////////////////////////////
// SIMULATION_INVOLVE macro
//NOTE: Assign to one of this for further computation
//   --> The value order are important here
//       Don't change it
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
//Na-transient   CHANNEL_NAT macro
#define NAT_HODGKIN_HUXLEY_1952 1
#define NAT_WOLF_2005           2
#define NAT_HAY_2011            3
#define NAT_SCHWEIGHOFER_1999   4
// Na-persistent CHANNE_NAP macro
#define NAP_WOLF_2005           2
//}}}
//{{{ K-models
// KAf   CHANNEL_KAf macro
#define KAf_WOLF_2005          2
// KAs   CHANNEL_KAs macro
#define KAs_WOLF_2005          2
// KIR   CHANNEL_KIR macro
#define KIR_WOLF_2005          2
// KRP   CHANNEL_KRP macro
#define KRP_WOLF_2005          2
// KDR   CHANNEL_KDR macro
#define KDR_HODGKIN_HUXLEY_1952 1
#define KDR_SCHWEIGHOFER_1999   4

// BK-alpha
// BK-alphabeta   CHANNEL_BKalphabeta macro
// NOTE: Those with the same values are indeed the same model
//        just being used in different papers
#define BKalphabeta_SHAO_1999       2       
#define BKalphabeta_WOLF_2005       2       

// SK       CHANNEL_SK macro
#define SK_MOCZYDLOWSKI_1993 2
#define SK_WOLF_2005    2
//}}}
//{{{ Ca-models
// CaL CHANNEL_CaL macro  (designed for maximal user's defined parameters)
#define CaL_GENERAL    100
// CaLv12 (HVA)   CHANNEL_CaLv12 macro
#define CaLv12_GHK_Standen_Stanfield_1982 3
#define CaLv12_GHK_WOLF_2005 2
// CaLv13 (LVA)  CHANNEL_CaLv13 macro
#define CaLv13_GHK_WOLF_2005 2
// CaN      CHANNEL_CaN macro
#define CaN_GHK_WOLF_2005 2
// CaPQ     CHANNEL_CaPQ macro
#define CaPQ_GHK_WOLF_2005 2
// CaR      CHANNEL_CaR macro
#define CaR_GHK_WOLF_2005 2
// CaT      CHANNEL_CaT macro
#define CaT_GHK_WOLF_2005 2
//}}}
//{{{ Synapse Receptors
// NMDAR      RECEPTOR_NMDA macro 
#define NMDAR_POINTPROCESS                    1
#define NMDAR_BEHABADI_2012                   2
#define NMDAR_JADI_2012                       3
#define NMDAR_JAHR_STEVENS_1990               4
// AMPAR      RECEPTOR_AMPA macro
#define AMPAR_POINTPROCESS                    1
#define AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994  3
// GABA_A     RECEPTOR_GABAA
#define GABAAR_POINTPROCESS 1
#define GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994  3
// GABA_B     RECEPTOR_GABAB
#define GABABR_DESTEXHE_SEJNOWSKI_1996  3
//}}}
//{{{ Sarcolema membrane exchanger/pump
// PMCA       PUMP_PMCA:
//{{{
#define PMCA_PUMPRATE_CONSTANT 1
#define PMCA_Traub_Llinas_1997 PMCA_PUMPRATE_CONSTANT  
#define PMCA_PUMPRATE_VOLTAGE_FUNCTION 2 
#define PMCA_Zador_Koch_Brown_1990 PMCA_PUMPRATE_VOLTAGE_FUNCTION 
#define PMCA_Jafri_Rice_Winslow_1998 3
#define PMCA_Greenstein_Winslow_2002       4
//}}}
// NCX        EXCHANGER_NCX
#define NCX_Gabbiani_Midtgaard_Kopfel_1994 2
#define NCX_Weber_Bers_2001  3
//}}}
//{{{ ER membrane channels/pump
// RYR       CHANNEL_RYR

// IP3R     CHANNEL_IP3R

// SERCA    PUMP_SERCA
#define SERCA_Tran_Crampin_2009  2

//}}}

//////////////////////////////////////////////////////////////////////
// Synaptic model design
//   SYNAPSE_MODEL_STRATEGY macro
#define USE_PRESYNAPTICPOINT   1
#define USE_SYNAPTICCLEFT      2
//{{{ Neurotransmitter Update method
// Neurotransmitter update method 
//   Use a continuous function to transform presynaptic Vm 
//   into transmitter concentration as a sigmoid function
//   from Tmin to Tmax
//   T(Vpre) = Tmin + Tmax / ( 1 + exp (- (Vpre - Vp) /  Kp) )
#define NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994 1
//}}}
///////////////////////////////////////////////////////////////////////
// How you want to model dynamics 
//{{{ Calcium-dynamics method
#define REGULAR_DYNAMICS  1
#define FAST_BUFFERING   2
//}}}

///////////////////////////////////////////////////////////////////////
/// Define what models are available here
//    MODEL_TO_USE macro
//{{{
#define _MODEL_NOT_DEFINED    0 
#define _MODEL_TESTING        1
#define _MSN_2005_WOLF        2 
#define _MSN_2016_TUAN_JAMES  3
//}}}
// define 


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// USER-SELECTED SECTION 
// 1. to choose a model: select the proper value for MODEL_TO_USE
#define MODEL_TO_USE _MSN_2005_WOLF
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////
// MODEL DESIGN
// IMPORTANT: try not to modify an existing one, instead create a new one
//           by copying an existing model, and define a new name
//           in section MODE_TO_USE macro 
//           user are free to work on section _MODEL_TESTING
// 2. configure each model
//  2.a select what compartmental variables to use
//  2.b to disable any channel from the model, just comment it out
#if MODEL_TO_USE == _MSN_2005_WOLF
//#define SYNAPSE_MODEL_STRATEGY USE_PRESYNAPTICPOINT
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
  #define SIMULATION_INVOLVE VMONLY
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING
//{{{
  #define CHANNEL_NAT NAT_WOLF_2005
  #define CHANNEL_NAP NAP_WOLF_2005
  #define CHANNEL_KAf KAf_WOLF_2005
  #define CHANNEL_KAs KAs_WOLF_2005
  #define CHANNEL_KIR KIR_WOLF_2005
  #define CHANNEL_KRP KRP_WOLF_2005
  #define CHANNEL_BKalphabeta  BKalphabeta_WOLF_2005
  #define CHANNEL_SK SK_WOLF_2005
  #define CHANNEL_CaLv12 CaLv12_GHK_WOLF_2005
  #define CHANNEL_CaLv13 CaLv13_GHK_WOLF_2005
  #define CHANNEL_CaN CaN_GHK_WOLF_2005
  #define CHANNEL_CaPQ CaPQ_GHK_WOLF_2005
  #define CHANNEL_CaR CaR_GHK_WOLF_2005
  #define CHANNEL_CaT CaT_GHK_WOLF_2005
  #define RECEPTOR_AMPA AMPAR_POINTPROCESS
  #define RECEPTOR_NMDA NMDAR_POINTPROCESS
//}}}

#elif MODEL_TO_USE == _MSN_2016_TUAN_JAMES
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
  #define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994

  #define SIMULATION_INVOLVE VM_CACYTO_CAER
  #define CALCIUM_CYTO_DYNAMICS REGULAR_DYNAMICS
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING 
//{{{
  #define CHANNEL_NAT NAT_WOLF_2005
  #define CHANNEL_NAP NAP_WOLF_2005
  #define CHANNEL_KAf KAf_WOLF_2005
  #define CHANNEL_KAs KAs_WOLF_2005
  #define CHANNEL_KIR KIR_WOLF_2005
  #define CHANNEL_KRP KRP_WOLF_2005
  #define CHANNEL_BKalphabeta  BKalphabeta_WOLF_2005
  #define CHANNEL_SK SK_WOLF_2005
  #define CHANNEL_CaLv12 CaLv12_GHK_WOLF_2005
  #define CHANNEL_CaLv13 CaLv13_GHK_WOLF_2005
  #define CHANNEL_CaN CaN_GHK_WOLF_2005
  #define CHANNEL_CaPQ CaPQ_GHK_WOLF_2005
  #define CHANNEL_CaR CaR_GHK_WOLF_2005
  #define CHANNEL_CaT CaT_GHK_WOLF_2005
  #define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define RECEPTOR_NMDA NMDAR_JAHR_STEVENS_1990 
  #define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define CHANNEL_RYR RYR_SOMETHING
  #define CHANNEL_IP3R  IP3R_SOMETHING
  #define EXCHANGER_NCX  NCX_Weber_Bers_2001
  #define PUMP_PMCA  PMCA_PUMPRATE_VOLTAGE_FUNCTION
  #define PUMP_SERCA  SERCA_Tran_Crampin_2009
//}}}

#else
  NOT IMPLEMENTED YET
#endif

//////////////////////////////////////////////////////////////////////
// Default setting
//
//{{{
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
#ifndef CHANNEL_CaL
#define CHANNEL_CaL _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_CaLv12
#define CHANNEL_CaLv12 _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_CaLv13
#define CHANNEL_CaLv13 _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_CaN
#define CHANNEL_CaN _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_CaPQ
#define CHANNEL_CaPQ _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_CaR
#define CHANNEL_CaR _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_CaT
#define CHANNEL_CaT _COMPONENT_UNDEFINED
#endif
#ifndef MODEL_TO_USE
#define MODEL_TO_USE _MODEL_NOT_DEFINED
#endif
#ifndef RECEPTOR_AMPA
#define RECEPTOR_AMPA _COMPONENT_UNDEFINED
#endif
#ifndef RECEPTOR_NMDA
#define RECEPTOR_NMDA _COMPONENT_UNDEFINED
#endif
#ifndef RECEPTOR_GABAA
#define RECEPTOR_GABAA _COMPONENT_UNDEFINED
#endif
#ifndef RECEPTOR_GABAB
#define RECEPTOR_GABAB _COMPONENT_UNDEFINED
#endif

#ifndef PUMP_PMCA
#define PUMP_PMCA _COMPONENT_UNDEFINED
#endif
#ifndef EXCHANGER_NCX
#define EXCHANGER_NCX _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_RYR
#define CHANNEL_RYR _COMPONENT_UNDEFINED
#endif
#ifndef CHANNEL_IP3R
#define CHANNEL_IP3R _COMPONENT_UNDEFINED
#endif
#ifndef PUMP_SERCA
#define PUMP_SERCA _COMPONENT_UNDEFINED
#endif
	 // implicit synapse space
#ifndef SYNAPSE_MODEL_STRATEGY
#define SYNAPSE_MODEL_STRATEGY USE_PRESYNAPTICPOINT
#endif

	 // if explicit synapse space is used
#if SYNAPSE_MODEL_STRATEGY == USE_SYNAPTICCLEFT 
	 // default: use simple estimation of neurotransmitter as 
	 // given in Dextexhe-Mainen-Sejnowski 1994 
#ifndef GLUTAMATE_UPDATE_METHOD
#define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#endif
#ifndef GABA_UPDATE_METHOD
#define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#endif
#endif

#ifndef SIMULATION_INVOLVE
#define SIMULATION_INVOLVE  VMONLY
#endif
#ifndef CALCIUM_CYTO_DYNAMICS
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
#endif
#ifndef CALCIUM_ER_DYNAMICS
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING
#endif

//}}}


#endif
