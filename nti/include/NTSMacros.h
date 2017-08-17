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
#define zR  8.314472e3 //[mJ/(K.mol)] - universal gas constant
#define zCa 2          // valance of Ca2+ ions
#define zNa 1          // valence of Na+ ions
#define zK  1          // valence of K+ ions
#define zkB  1.381e-23  // [J/K] = Joule/Kelvin = Boltzmann constant (R/N_A)
#define zN_A 6.022e23   // [1/mol] = number of molecuels/atoms/ions per mole - Avogadro number

#define zCa2F2_R ((zCa*zCa)*(zF*zF)/(zR))
#define zCaF_R (zCa*zF/(zR))
#define R_zCaF (zR/(zCa*zF))
#define zF_RT (zF / (zR * *getSharedMembers().T))
#define mM2uM 1e3   // conversion factor
#define uM2mM 1e-3  // conversion factor
#define MIN_RESISTANCE_VALUE 0.0001  //[GOhm*um] - keep this for numerical stability

#define AvogN  (6.0223*1e23)     // Avogadro number

#define TKelvin 273.15          //[celcius] - absolute temperature
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
// {{{to be removed
//NOTE: Assign to one of this for further computation
//   --> The value order are important here
//       Don't change it
//{{{
//#define VMONLY 1
//#define VM_CACYTO 2
//#define VM_CACYTO_CAER 3
//#define VM_CACYTO_CAER_DOPA 4
// }}}
//}}}
//IMPORTANT: Enable one or many from this list in your cell model
//          SIMULATE_VM must always be defined
#define SIMULATE_VM
//#define SIMULATE_CACYTO
//#define SIMULATE_CAER
//#define SIMULATE_DOPA
//#define SIMULATE_IP3
 //default


///////////////////////////////////////////////////////////////////////
// list of paper models
#define _COMPONENT_UNDEFINED    0
//{{{
//{{{ Na-models
//Na-transient   CHANNEL_NAT macro
#define NAT_HODGKIN_HUXLEY_1952 1
#define NAT_WOLF_2005           2
#define NAT_HAY_2011            3
#define NAT_SCHWEIGHOFER_1999   4
#define NAT_COLBERT_PAN_2002    5
#define NAT_TRAUB_1994          6
#define NAT_OGATA_TATEBAYASHI_1990  7
#define NAT_WANG_BUSZAKI_1996  8
#define NAT_MAHON_2000         9

#define _NAT_DEFAULT NAT_HODGKIN_HUXLEY_1952

//NAT_AIS Channel Channel_NAT_AIS macro
#define NAT_AIS_TRAUB_1994     1     

#define _NAT_AIS_DEFAULT NAT_AIS_TRAUB_1994
// Na-persistent CHANNEL_NAP macro
#define NAP_WOLF_2005           2
#define NAP_MAGISTRETTI_1999    3
#define _NAP_DEFAULT NAP_WOLF_2005

#define NAS_MAHON_2000  2

#define _NAS_DEFAULT NAS_MAHON_2000

//}}}

//{{{ K-models
// KAf   CHANNEL_KAf macro
//{{{
#define KAf_TRAUB_1994             2
#define KAf_MAHON_2000             3
#define KAf_KORNGREEN_SAKMANN_2000 4
#define KAf_WOLF_2005              5
#define KAf_EVANS_2012             6

#define _KAf_DEFAULT           KAf_WOLF_2005
//}}}
// KAs   CHANNEL_KAs macro
//{{{
#define KAs_MAHON_2000             2 
#define KAs_KORNGREEN_SAKMANN_2000 3
#define KAs_WOLF_2005              4

#define _KAs_DEFAULT           KAs_WOLF_2005
//}}}
// KIR   CHANNEL_KIR macro
//{{{
#define KIR_MAHON_2000         2
#define KIR_WOLF_2005          3

#define _KIR_DEFAULT           KIR_WOLF_2005
//}}}
// KRP   CHANNEL_KRP macro
//{{{
#define KRP_MAHON_2000         2 
#define KRP_WOLF_2005          3

#define _KRP_DEFAULT           KRP_WOLF_2005
//}}}
// KDR   CHANNEL_KDR macro
//{{{
#define KDR_HODGKIN_HUXLEY_1952 2
#define KDR_TRAUB_1994          3
#define KDR_TRAUB_1995          4
#define KDR_WANG_BUSZAKI_1996   5
#define KDR_SCHWEIGHOFER_1999   6
#define KDR_MAHON_2000          7

#define _KDR_DEFAULT KDR_TRAUB_1994

// KDR_AIS  CHANNEL_KDR_AIS macro
#define KDR_AIS_TRAUB_1994  1
#define KDR_AIS_TRAUB_1995  1

#define _KDR_AIS_DEFAULT KDR_AIS_TRAUB_1994
//}}}
// BK-alpha
//{{{
// BK-alphabeta   CHANNEL_BKalphabeta macro
// NOTE: Those with the same values are indeed the same model
//        just being used in different papers
#define BKalphabeta_SHAO_1999       2       
#define BKalphabeta_WOLF_2005       BKalphabeta_SHAO_1999       

#define _BKalphabeta_DEFAULT        BKalphabeta_SHAO_1999
// BK
// BK_ CHANNEL_BK macro
#define BK_TRAUB_1994 1

#define _BK_DEFAULT BK_TRAUB_1994
//}}}
// SK       CHANNEL_SK macro
//{{{
#define SK_MOCZYDLOWSKI_1993 2
#define SK_WOLF_2005    2
#define SK2_KOHLER_ADELMAN_1996_RAT 3
#define SK1_KOHLER_ADELMAN_1996_HUMAN 4
#define SK_TRAUB_1994 5

#define _SK_DEFAULT SK_WOLF_2005
//}}}
// MK (Muscarinic-sensitive K+ current) CHANNEL_MK macro
//{{{
#define  MK_ADAMS_BROWN_CONSTANTI_1982  1

#define _MK_DEFAULT MK_ADAMS_BROWN_CONSTANTI_1982
//}}}

// Kv31 (Shaker-related)
//{{{
#define  Kv31_RETTIG_1992  1

#define _Kv31_DEFAULT Kv31_RETTIG_1992
//}}}
//}}}

//{{{ HCN-models
// HCN   CHANNEL_HCN macro
#define HCN_HUGUENARD_MCCORMICK_1992 1
#define HCN_VANDERGIESSEN_DEZEEUW_2008 2
#define HCN_KOLE_2006 3
#define HCN_HAY_2011 4

#define _HCN_DEFAULT HCN_HUGUENARD_MCCORMICK_1992
//}}}

//{{{ Ca-models
//CaHVA  CHANNEL_CaHVA
//{{{
#define CaHVA_TRAUB_1994  1
#define CaHVA_REUVENI_AMITAI_GUTNICK_1993 2

#define _CaHVA_DEFAULT CaHVA_TRAUB_1994
//}}}
//CaLVA  CHANNEL_CaLVA
//{{{
#define CaLVA_AVERY_JOHNSTON_1996 2
#define CaLVA_HAY_2011 3

#define _CaLVA_DEFAULT CaLVA_HAY_2011 
//}}}
// CaL CHANNEL_CaL macro  (designed for maximal user's defined parameters)
//{{{
//}}}
#define CaL_GENERAL    100
// CaLv12 (HVA)   CHANNEL_CaLv12 macro
//{{{
#define CaLv12_GHK_Standen_Stanfield_1982_option1 3
#define CaLv12_GHK_Standen_Stanfield_1982_option2 4
#define CaLv12_GHK_WOLF_2005 2

#define _CaLv12_DEFAULT      CaLv12_GHK_WOLF_2005
//}}}
// CaLv13 (HVA)  CHANNEL_CaLv13 macro
//{{{
#define CaLv13_GHK_WOLF_2005 2

#define _CaLv13_DEFAULT    CaLv13_GHK_WOLF_2005
//}}}
// CaN      CHANNEL_CaN macro
//{{{
#define CaN_GHK_WOLF_2005 2

#define _CaN_DEFAULT   CaN_GHK_WOLF_2005
//}}}
// CaPQ     CHANNEL_CaPQ macro
//{{{
#define CaPQ_GHK_WOLF_2005 2

#define _CaPQ_DEFAULT   CaPQ_GHK_WOLF_2005
//}}}
// CaR      CHANNEL_CaR macro
//{{{
#define CaR_GHK_WOLF_2005 2

#define _CaR_DEFAULT  CaR_GHK_WOLF_2005
//}}}
// CaT (LVA)      CHANNEL_CaT macro
//{{{
#define CaT_GHK_WOLF_2005 2

#define _CaT_DEFAULT CaT_GHK_WOLF_2005
//}}}
//}}}

//{{{ Synapse Receptors
// NMDAR      RECEPTOR_NMDA macro 
#define NMDAR_POINTPROCESS                    1
#define NMDAR_BEHABADI_2012                   2
#define NMDAR_JADI_2012                       3
#define NMDAR_JAHR_STEVENS_1990               4
#define NMDAR_BEHABADI_2012_MODIFIED          5
//#define NMDAR_Markov_DESTEXHE_MAINEN_SEJNOWSKI_1994     6

#define _NMDAR_DEFAULT   NMDAR_POINTPROCESS
// AMPAR      RECEPTOR_AMPA macro
#define AMPAR_POINTPROCESS                    1
#define AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994  3
//#define AMPAR_Markov_DESTEXHE_MAINEN_SEJNOWSKI_1994  4

#define _AMPAR_DEFAULT AMPAR_POINTPROCESS
// GABA_A     RECEPTOR_GABAA
#define GABAAR_POINTPROCESS 1
#define GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994  3

#define _GABAAR_DEFAULT GABAAR_POINTPROCESS
// GABA_B     RECEPTOR_GABAB
#define GABABR_DESTEXHE_SEJNOWSKI_1996  3

#define _GABABR_DEFAULT GABABR_DESTEXHE_SEJNOWSKI_1996
//}}}

//{{{ Sarcolema membrane exchanger/pump
// PMCA       PUMP_PMCA:
//{{{
  // PUMPRATE_CONSTANT = all branches have the same clearance rate
#define PMCA_PUMPRATE_CONSTANT 1
  // each branch has a different clearance rate
#define PMCA_PUMPRATE_CONSTANT_DYNAMICS 2
#define PMCA_Traub_Llinas_1997 PMCA_PUMPRATE_CONSTANT  
#define PMCA_PUMPRATE_VOLTAGE_FUNCTION 3 
#define PMCA_Zador_Koch_Brown_1990 PMCA_PUMPRATE_VOLTAGE_FUNCTION 
#define PMCA_Jafri_Rice_Winslow_1998 4
#define PMCA_Greenstein_Winslow_2002       5

#define _PMCA_DEFAULT PMCA_PUMPRATE_CONSTANT
//}}}
// NCX        EXCHANGER_NCX
#define NCX_Gabbiani_Midtgaard_Kopfel_1994 2
#define NCX_Weber_Bers_2001  3

#define _NCX_DEFAULT  NCX_Weber_Bers_2001
//}}}

//{{{ ER membrane channels/pump
// RYR       CHANNEL_RYR
#define RYR2_WILLIAMS_JAFRI_2011 1

#define _RYR_DEFAULT RYR2_WILLIAMS_JAFRI_2011

// IP3R     CHANNEL_IP3R
#define IP3R_ULLAH_MAK_PEARSON_2012 1
#define IP3R_LI_RINZEL_1994         2
#define IP3R_SMITH_2002             3
#define IP3R_SNEYD_DUFOUR_2002      4

#define _IP3R_DEFAULT IP3R_ULLAH_MAK_PEARSON_2012

// SERCA    PUMP_SERCA
#define SERCA_Klein_Schneider_1991 1
#define SERCA_Tran_Crampin_2009  2

#define _SERCA_DEFAULT SERCA_Tran_Crampin_2009

//}}}
//}}}

//////////////////////////////////////////////////////////////////////
// Synaptic model design
//   SYNAPSE_MODEL_STRATEGY macro
#define USE_PRESYNAPTICPOINT   1
#define USE_SYNAPTICCLEFT      2
//{{{ Neurotransmitter Update method
//     GLUTAMATE_UPDATE_METHOD 
//     GABA_UPDATE_METHOD NEURO
// Neurotransmitter update method 
//   Use a continuous function to transform presynaptic Vm 
//   into transmitter concentration as a sigmoid function
//   from Tmin to Tmax
//   T(Vpre) = Tmin + Tmax / ( 1 + exp (- (Vpre - Vp) /  Kp) )
#define NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994 1
#define NEUROTRANSMITTER_BIEXPONENTIAL 2
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
//  NOTE: Code-based is 
//     1xx ~ Pyramidal
//     2xx ~ MSN
//     3xx ~ Interneuron
//     4xx ~ IO                  
//     5xx ~ spines
//     6xx ~ microdomain
//{{{
#define _MODEL_NOT_DEFINED    0 
#define _MODEL_TESTING        1

#define _PYRAMIDAL_2011_HAY  100
#define _PYRAMIDAL_L5b_2016_TUAN_JAMES 101
#define _PYRAMIDAL_L5b_2017_TUAN_JAMES 102

#define _MSN_2005_WOLF            200    
#define _MSN_2016_TUAN_JAMES      201
#define _MSN_2012_EVANS_BLACKWELL 202
#define _MSN_2000_MAHON           203

#define _INFERIOR_OLIVE_1999_SCHWEIGHOFER 400

#define _INTERNEURON_TRAUB_1995 300
//TODO: #define _INTERNEURON_WANG_BUSZAKI_1996 301

#define _SPINE_MSN_2017_TUAN_JAMES 500

#define _MICRODOMAIN_DA_NEURON_2017_TUAN_JAMES 600
#define _MICRODOMAIN_MSN_STRIATUM_NEURON_2017_TUAN_JAMES 601
//}}}
// define 


///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// USER-SELECTED SECTION 
// 1. to choose a model: select the proper value for MODEL_TO_USE
// Please don't do it here, edit inside this file
#include "Model2Use.h"
///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////
// MODEL DESIGN
// IMPORTANT: try not to modify an existing one, instead create a new one
//           by copying an existing model, and define a new name
//           in section MODEL_TO_USE macro 
//           user are free to work on section _MODEL_TESTING
// 2. configure each model
//  2.a select what compartmental variables to use
//  2.b to disable any channel from the model, just comment it out
#if MODEL_TO_USE == _MSN_2000_MAHON
//{{{
  //#define SYNAPSE_MODEL_STRATEGY USE_PRESYNAPTICPOINT
  #define SIMULATE_VM
  //#define SIMULATE_CACYTO
  //#define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
#define DEBUG_COMPARTMENT
#define USE_SOMA_AS_POINT
#define WRITE_GATES

//{{{
  #define CHANNEL_NAT NAT_MAHON_2000
  #define CHANNEL_NAP NAP_MAHON_2000
  #define CHANNEL_NAS NAS_MAHON_2000

  #define CHANNEL_KIR KIR_MAHON_2000
  #define CHANNEL_KDR KDR_MAHON_2000
  #define CHANNEL_KAf KAf_MAHON_2000 
  #define CHANNEL_KAs KAs_MAHON_2000
  #define CHANNEL_KRP KRP_MAHON_2000

//}}}
//}}}
#elif   MODEL_TO_USE == _MSN_2005_WOLF
//{{{
#define STRETCH_SOMA_WITH 105 //#135.0    
#define SCALING_NECK_FROM_SOMA 5.9  //>1: make neck smaller
//#define SCALING_NECK_FROM_SOMA 7.0  //>1: make neck smaller
//#define SCALING_NECK_FROM_SOMA 6.5  //>1: make neck smaller
//#define SCALING_NECK_FROM_SOMA 6.2  //>1: make neck smaller
//#define SCALING_NECK_FROM_SOMA 6.15  //>1: make neck smaller
//#define SCALING_NECK_FROM_SOMA 6.0  //>1: make neck smaller
//#define SCALING_NECK_FROM_SOMA 5.0  //>1: make neck smaller
//#define SCALING_NECK_FROM_SOMA 4.5  //>1: make neck smaller
//#define SYNAPSE_MODEL_STRATEGY USE_PRESYNAPTICPOINT
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
//  #define SIMULATION_INVOLVE VM_CACYTO
#define SIMULATE_VM
#define SIMULATE_CACYTO
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING
//{{{ list channels 
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

  #define PUMP_PMCA PMCA_PUMPRATE_CONSTANT_DYNAMICS

  #define RECEPTOR_AMPA AMPAR_POINTPROCESS
  #define RECEPTOR_NMDA NMDAR_POINTPROCESS
  #define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
//}}}
//}}}
#elif MODEL_TO_USE == _INTERNEURON_TRAUB_1995
//{{{
#define STRETCH_SOMA_WITH 0.0    //seem ok  value if used
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
//  #define SIMULATION_INVOLVE VM_CACYTO
#define SIMULATE_VM
#define SIMULATE_CACYTO
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING

//{{{ list channels 
  #define CHANNEL_NAT NAT_TRAUB_1994
  #define CHANNEL_NAT_AIS NAT_AIS_TRAUB_1994
  #define CHANNEL_KDR KDR_TRAUB_1995
  #define CHANNEL_KDR_AIS KDR_AIS_TRAUB_1995
  #define CHANNEL_KAf KAf_TRAUB_1994
  #define CHANNEL_BK BK_TRAUB_1994
  #define CHANNEL_SK SK_TRAUB_1994
  #define CHANNEL_CaHVA CaHVA_TRAUB_1994
  #define PUMP_PMCA  PMCA_PUMPRATE_CONSTANT_DYNAMICS

//}}}

//}}}
#elif MODEL_TO_USE == _MSN_2016_TUAN_JAMES
#define SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM 
//{{{
//#define STRETCH_SOMA_WITH 105 //#135.0    
#define SCALING_NECK_FROM_SOMA 5.9  //>1: make neck smaller
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
  #define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define  IDEA_CURRENTONCOMPT 

#define SIMULATE_VM
#define SIMULATE_CACYTO
//#define SIMULATE_CAER
//#define SIMULATE_IP3
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING 
  #define IP3_CYTO_DYNAMICS  REGULAR_DYNAMICS
//{{{//list channels
  #define CHANNEL_NAT NAT_WOLF_2005
  //#define CHANNEL_NAT NAT_OGATA_TATEBAYASHI_1990
  #define CHANNEL_NAP NAP_WOLF_2005
  #define CHANNEL_KAf KAf_WOLF_2005
  //#define CHANNEL_KAf KAf_EVANS_2012
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

  #define EXCHANGER_NCX  NCX_Weber_Bers_2001
  #define PUMP_PMCA  PMCA_PUMPRATE_CONSTANT_DYNAMICS
  //NOTE: When switching to the below model
  //we no longer use 'tau' but Ipmcabar
  // which need tobe updated in the ChanParams.par
  //#define PUMP_PMCA  PMCA_Jafri_Rice_Winslow_1998

  #define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define RECEPTOR_NMDA NMDAR_JAHR_STEVENS_1990 
  #define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994

  #define CHANNEL_RYR RYR2_WILLIAMS_JAFRI_2011
  //#define CHANNEL_IP3R  IP3R_ULLAH_MAK_PEARSON_2012
  #define CHANNEL_IP3R  IP3R_LI_RINZEL_1994
  #define PUMP_SERCA  SERCA_Tran_Crampin_2009
//}}}

//}}}
#elif MODEL_TO_USE == _MSN_2012_EVANS_BLACKWELL
#define SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM 
//{{{
//#define STRETCH_SOMA_WITH 105 //#135.0    
//#define SCALING_NECK_FROM_SOMA 5.9  //>1: make neck smaller
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
  #define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994

#define SIMULATE_VM
#define SIMULATE_CACYTO
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING 
  #define IP3_CYTO_DYNAMICS  REGULAR_DYNAMICS
//{{{//list channels
  #define CHANNEL_NAT NAT_OGATA_TATEBAYASHI_1990
  //notuse#define CHANNEL_NAP NAP_WOLF_2005
  #define CHANNEL_KAf KAf_EVANS_2012
  //#define CHANNEL_KAs KAs_WOLF_2005
  //#define CHANNEL_KIR KIR_WOLF_2005
  //#define CHANNEL_KRP KRP_WOLF_2005
  //#define CHANNEL_BKalphabeta  BKalphabeta_WOLF_2005
  //#define CHANNEL_SK SK_WOLF_2005
  //#define CHANNEL_CaLv12 CaLv12_GHK_WOLF_2005
  //#define CHANNEL_CaLv13 CaLv13_GHK_WOLF_2005
  //#define CHANNEL_CaN CaN_GHK_WOLF_2005
  //#define CHANNEL_CaPQ CaPQ_GHK_WOLF_2005
  //#define CHANNEL_CaR CaR_GHK_WOLF_2005
  //#define CHANNEL_CaT CaT_GHK_WOLF_2005

  //#define EXCHANGER_NCX  NCX_Weber_Bers_2001
  //#define PUMP_PMCA  PMCA_PUMPRATE_CONSTANT_DYNAMICS
  ////NOTE: When switching to the below model
  ////we no longer use 'tau' but Ipmcabar
  //// which need tobe updated in the ChanParams.par
  ////#define PUMP_PMCA  PMCA_Jafri_Rice_Winslow_1998

  //#define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define RECEPTOR_NMDA NMDAR_JAHR_STEVENS_1990 
  //#define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994

  //#define CHANNEL_RYR RYR2_WILLIAMS_JAFRI_2011
  ////#define CHANNEL_IP3R  IP3R_ULLAH_MAK_PEARSON_2012
  //#define CHANNEL_IP3R  IP3R_LI_RINZEL_1994
  //#define PUMP_SERCA  SERCA_Tran_Crampin_2009
//}}}

//}}}
#elif MODEL_TO_USE == _PYRAMIDAL_2011_HAY
//{{{
  //#define SYNAPSE_MODEL_STRATEGY USE_PRESYNAPTICPOINT
#define STRETCH_SOMA_WITH 35.0    //seem ok  value if used

#define SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM 
  #define IDEA_DYNAMIC_INITIALVOLTAGE
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
  #define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define  IDEA_CURRENTONCOMPT 

#define SIMULATE_VM
#define SIMULATE_CACYTO
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING
//{{{ list channels
  #define CHANNEL_NAT NAT_HAY_2011
  #define CHANNEL_NAP NAP_MAGISTRETTI_1999
  #define CHANNEL_HCN HCN_HAY_2011
  #define CHANNEL_MK  MK_ADAMS_BROWN_CONSTANTI_1982
  #define CHANNEL_KAf KAf_KORNGREEN_SAKMANN_2000
  #define CHANNEL_KAs KAs_KORNGREEN_SAKMANN_2000
  #define CHANNEL_Kv31 Kv31_RETTIG_1992
  #define CHANNEL_SK SK2_KOHLER_ADELMAN_1996_RAT
  #define CHANNEL_CaHVA CaHVA_REUVENI_AMITAI_GUTNICK_1993
  #define CHANNEL_CaLVA CaLVA_HAY_2011
  #define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define RECEPTOR_NMDA NMDAR_JAHR_STEVENS_1990
  //#define RECEPTOR_NMDA NMDAR_JADI_2012
  #define RECEPTOR_NMDA NMDAR_BEHABADI_2012
  #define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define PUMP_PMCA PMCA_PUMPRATE_VOLTAGE_FUNCTION  
  #define PUMP_PMCA PMCA_PUMPRATE_CONSTANT_DYNAMICS
  //#define PUMP_PMCA  PMCA_Zador_Koch_Brown_1990 
  //#define PUMP_PMCA  PMCA_Jafri_Rice_Winslow_1998
//}}}

//}}}
#elif MODEL_TO_USE == _PYRAMIDAL_L5b_2016_TUAN_JAMES
//{{{
//#define STRETCH_SOMA_WITH 0.0    // NOT WORKING
//#define STRETCH_SOMA_WITH 10.0    //seem ok  value if used
#define STRETCH_SOMA_WITH 20.0    // GOOD VALUE - time2AP is long enough so that AP can stop
             //after the 5ms stim; and then can end properly without generating the doublets
//#define STRETCH_SOMA_WITH 30.0    //seem ok  value if used
//#define STRETCH_SOMA_WITH 35.0    // (DEFAULT) working to trigger AP 
  //#define SCALING_NECK_FROM_SOMA 2.0  //>1: make neck smaller
  #define SCALING_NECK_FROM_SOMA 2.2  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 2.35  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 3.0  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 4.0  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 8.0  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 6.0  //>1: make neck smaller
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
  //#define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define GLUTAMATE_UPDATE_METHOD  NEUROTRANSMITTER_BIEXPONENTIAL
  #define GABA_UPDATE_METHOD      NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define GABA_UPDATE_METHOD       NEUROTRANSMITTER_BIEXPONENTIAL

  //#define SIMULATION_INVOLVE VM_CACYTO
#define SIMULATE_VM
#define SIMULATE_CACYTO
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING
//{{{ list channels
  #define CHANNEL_NAT NAT_HAY_2011
  //#define CHANNEL_NAT NAT_COLBERT_PAN_2002
  //#define CHANNEL_NAT NAT_WOLF_2005
  #define CHANNEL_NAP NAP_MAGISTRETTI_1999
  //#define CHANNEL_NAP NAP_WOLF_2005
  #define CHANNEL_HCN HCN_HAY_2011
  #define CHANNEL_MK  MK_ADAMS_BROWN_CONSTANTI_1982
  #define CHANNEL_KAf KAf_KORNGREEN_SAKMANN_2000
  #define CHANNEL_KAs KAs_KORNGREEN_SAKMANN_2000
  //#define CHANNEL_KAf KAf_WOLF_2005
  //#define CHANNEL_KAs KAs_WOLF_2005
  //#define CHANNEL_KIR KIR_WOLF_2005
  //#define CHANNEL_KRP KRP_WOLF_2005
  //#define CHANNEL_BKalphabeta  BKalphabeta_WOLF_2005
  #define CHANNEL_Kv31 Kv31_RETTIG_1992
  #define CHANNEL_SK SK2_KOHLER_ADELMAN_1996_RAT
  #define CHANNEL_CaHVA CaHVA_REUVENI_AMITAI_GUTNICK_1993
  #define CHANNEL_CaLVA CaLVA_HAY_2011
  //#define CHANNEL_CaLv12 CaLv12_GHK_WOLF_2005
  //#define CHANNEL_CaLv13 CaLv13_GHK_WOLF_2005
  //#define CHANNEL_CaN CaN_GHK_WOLF_2005
  //#define CHANNEL_CaPQ CaPQ_GHK_WOLF_2005
  //#define CHANNEL_CaR CaR_GHK_WOLF_2005
  //#define CHANNEL_CaT CaT_GHK_WOLF_2005
  //#define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define RECEPTOR_AMPA AMPAR_Markov_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define RECEPTOR_NMDA NMDAR_JAHR_STEVENS_1990
  //#define RECEPTOR_NMDA NMDAR_JADI_2012
  #define RECEPTOR_NMDA NMDAR_BEHABADI_2012
  //#define RECEPTOR_NMDA NMDAR_BEHABADI_2012_MODIFIED
  #define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define PUMP_PMCA PMCA_PUMPRATE_VOLTAGE_FUNCTION  
  #define PUMP_PMCA PMCA_PUMPRATE_CONSTANT_DYNAMICS
  //#define PUMP_PMCA  PMCA_Zador_Koch_Brown_1990 
  //#define PUMP_PMCA  PMCA_Jafri_Rice_Winslow_1998
//}}}

//}}}
#elif MODEL_TO_USE == _PYRAMIDAL_L5b_2017_TUAN_JAMES
//{{{
//NOTE: This model incorporate CaER, IP3R
//#define STRETCH_SOMA_WITH 0.0    // NOT WORKING
//#define STRETCH_SOMA_WITH 10.0    //seem ok  value if used
#define STRETCH_SOMA_WITH 20.0    // GOOD VALUE - time2AP is long enough so that AP can stop
             //after the 5ms stim; and then can end properly without generating the doublets
//#define STRETCH_SOMA_WITH 30.0    //seem ok  value if used
//#define STRETCH_SOMA_WITH 35.0    // (DEFAULT) working to trigger AP 
  //#define SCALING_NECK_FROM_SOMA 2.0  //>1: make neck smaller
  #define SCALING_NECK_FROM_SOMA 2.2  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 2.35  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 3.0  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 4.0  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 8.0  //>1: make neck smaller
  //#define SCALING_NECK_FROM_SOMA 6.0  //>1: make neck smaller
  #define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
  //#define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define GLUTAMATE_UPDATE_METHOD  NEUROTRANSMITTER_BIEXPONENTIAL
  #define GABA_UPDATE_METHOD      NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define GABA_UPDATE_METHOD       NEUROTRANSMITTER_BIEXPONENTIAL

  //#define SIMULATION_INVOLVE VM_CACYTO
#define SIMULATE_VM
#define SIMULATE_CACYTO
#define SIMULATE_CAER
#define SIMULATE_IP3
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING
  #define IP3_CYTO_DYNAMICS  REGULAR_DYNAMICS
//{{{ list channels
  #define CHANNEL_NAT NAT_HAY_2011
  #define CHANNEL_NAP NAP_MAGISTRETTI_1999
  #define CHANNEL_HCN HCN_HAY_2011
  #define CHANNEL_MK  MK_ADAMS_BROWN_CONSTANTI_1982
  #define CHANNEL_KAf KAf_KORNGREEN_SAKMANN_2000
  #define CHANNEL_KAs KAs_KORNGREEN_SAKMANN_2000
  #define CHANNEL_Kv31 Kv31_RETTIG_1992
  #define CHANNEL_SK SK2_KOHLER_ADELMAN_1996_RAT
  #define CHANNEL_CaHVA CaHVA_REUVENI_AMITAI_GUTNICK_1993
  #define CHANNEL_CaLVA CaLVA_HAY_2011
  #define CHANNEL_IP3R IP3R_ULLAH_MAK_PEARSON_2012
  #define CHANNEL_RYR  RYR2_WILLIAMS_JAFRI_2011
  //#define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  #define RECEPTOR_AMPA AMPAR_Markov_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define RECEPTOR_NMDA NMDAR_JAHR_STEVENS_1990
  //#define RECEPTOR_NMDA NMDAR_JADI_2012
  #define RECEPTOR_NMDA NMDAR_BEHABADI_2012
  //#define RECEPTOR_NMDA NMDAR_BEHABADI_2012_MODIFIED
  #define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
  //#define PUMP_PMCA PMCA_PUMPRATE_VOLTAGE_FUNCTION  
  #define PUMP_PMCA PMCA_PUMPRATE_CONSTANT_DYNAMICS
  //#define PUMP_PMCA  PMCA_Zador_Koch_Brown_1990 
  //#define PUMP_PMCA  PMCA_Jafri_Rice_Winslow_1998
//}}}

//}}}
#elif MODEL_TO_USE == _INFERIOR_OLIVE_1999_SCHWEIGHOFER
//{{{
  #define IDEA_DYNAMIC_INITIALVOLTAGE
  #define SYNAPSE_MODEL_STRATEGY USE_PRESYNAPTICPOINT
#define SIMULATE_VM
#define SIMULATE_CACYTO
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
//{{{
  #define CHANNEL_NAT NAT_SCHWEIGHOFER_1999 //ok perfect
  //#define CHANNEL_NAT NAT_HODGKIN_HUXLEY_1952 //ok yet soma activation is lower
  //#define CHANNEL_NAT NAT_HAY_2011 //ok
  #define CHANNEL_HCN HCN_HUGUENARD_MCCORMICK_1992 
  //#define CHANNEL_KDR KDR_SCHWEIGHOFER_1999 //failed 
  //#define CHANNEL_KDR KDR_HODGKIN_HUXLEY_1952 
  #define CHANNEL_KDR  KDR_TRAUB_1994 //ok perfect
  #define CHANNEL_CaL CaL
  #define CHANNEL_CaH CaHVA_TRAUB_1994
  //#define PUMP_PMCA PMCA_PUMPRATE_CONSTANT
  #define PUMP_PMCA PMCA_PUMPRATE_CONSTANT_DYNAMICS
//}}}
//}}}
#elif MODEL_TO_USE == _SPINE_MSN_2017_TUAN_JAMES
//#{{{
#define SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM 
#define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
//#define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_BIEXPONENTIAL
//#define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define GABA_UPDATE_METHOD NEUROTRANSMITTER_BIEXPONENTIAL
#define  IDEA_CURRENTONCOMPT 
#define SIMULATE_VM
#define SIMULATE_CACYTO
#define SIMULATE_CAER
#define SIMULATE_IP3
#define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
#define CALCIUM_ER_DYNAMICS FAST_BUFFERING 
#define IP3_CYTO_DYNAMICS  REGULAR_DYNAMICS
#define TOUCHDETECT_SINGLENEURON_SPINES
//#define MICRODOMAIN_CALCIUM
//{{{//list channels
//#define CHANNEL_NAT NAT_WOLF_2005
#define CHANNEL_NAT NAT_OGATA_TATEBAYASHI_1990
//#define CHANNEL_NAT NAT_OGATA_TATEBAYASHI_1990
#define CHANNEL_NAP NAP_WOLF_2005
#define CHANNEL_KAf KAf_WOLF_2005
//#define CHANNEL_KAf KAf_EVANS_2012
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

#define EXCHANGER_NCX  NCX_Weber_Bers_2001
#define PUMP_PMCA  PMCA_PUMPRATE_CONSTANT_DYNAMICS
//NOTE: When switching to the below model
//we no longer use 'tau' but Ipmcabar
// which need tobe updated in the ChanParams.par
//#define PUMP_PMCA  PMCA_Jafri_Rice_Winslow_1998

#define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define RECEPTOR_NMDA NMDAR_JADI_2012
#define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994

#define CHANNEL_RYR RYR2_WILLIAMS_JAFRI_2011
//#define CHANNEL_IP3R  IP3R_ULLAH_MAK_PEARSON_2012
#define CHANNEL_IP3R  IP3R_LI_RINZEL_1994
#define PUMP_SERCA  SERCA_Tran_Crampin_2009
//}}}

//#}}}
#elif MODEL_TO_USE == _MICRODOMAIN_DA_NEURON_2017_TUAN_JAMES
//#{{{
#define SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM 
#define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
#define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define IDEA_CURRENTONCOMPT 
#define SIMULATE_VM
#define SIMULATE_CACYTO
#define SIMULATE_CAER
//#define SIMULATE_IP3
#define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
#define CALCIUM_ER_DYNAMICS FAST_BUFFERING 
  #define IP3_CYTO_DYNAMICS  REGULAR_DYNAMICS
//{{{//list channels
#define CHANNEL_NAT NAT_WOLF_2005
//#define CHANNEL_NAT NAT_OGATA_TATEBAYASHI_1990
#define CHANNEL_NAP NAP_WOLF_2005
#define CHANNEL_KAf KAf_WOLF_2005
//#define CHANNEL_KAf KAf_EVANS_2012
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

#define EXCHANGER_NCX  NCX_Weber_Bers_2001
#define PUMP_PMCA  PMCA_PUMPRATE_CONSTANT_DYNAMICS
//NOTE: When switching to the below model
//we no longer use 'tau' but Ipmcabar
// which need tobe updated in the ChanParams.par
//#define PUMP_PMCA  PMCA_Jafri_Rice_Winslow_1998

#define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define RECEPTOR_NMDA NMDAR_JAHR_STEVENS_1990 
#define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994

#define CHANNEL_RYR RYR2_WILLIAMS_JAFRI_2011
//#define CHANNEL_IP3R  IP3R_ULLAH_MAK_PEARSON_2012
#define CHANNEL_IP3R  IP3R_LI_RINZEL_1994
#define PUMP_SERCA  SERCA_Tran_Crampin_2009
//}}}

//#}}}
#elif MODEL_TO_USE == _MICRODOMAIN_MSN_STRIATUM_NEURON_2017_TUAN_JAMES
//#{{{
#define SUPPORT_DEFINING_SPINE_HEAD_N_NECK_VIA_PARAM 
#define SYNAPSE_MODEL_STRATEGY USE_SYNAPTICCLEFT
#define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define  IDEA_CURRENTONCOMPT 
#define SIMULATE_VM
#define SIMULATE_CACYTO
//#define SIMULATE_CAER
//#define SIMULATE_IP3
#define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
#define CALCIUM_ER_DYNAMICS FAST_BUFFERING 
  #define IP3_CYTO_DYNAMICS  REGULAR_DYNAMICS
//#define SCALING_NECK_FROM_SOMA 0.20  //>1: make neck smaller

#define TOUCHDETECT_SINGLENEURON_SPINES
#define MICRODOMAIN_CALCIUM
//{{{//list channels
//#define CHANNEL_NAT NAT_WOLF_2005
#define CHANNEL_NAT NAT_OGATA_TATEBAYASHI_1990
#define CHANNEL_NAP NAP_WOLF_2005
#define CHANNEL_KAf KAf_WOLF_2005
//#define CHANNEL_KAf KAf_EVANS_2012
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

#define EXCHANGER_NCX  NCX_Weber_Bers_2001
#define PUMP_PMCA  PMCA_PUMPRATE_CONSTANT_DYNAMICS
//NOTE: When switching to the below model
//we no longer use 'tau' but Ipmcabar
// which need tobe updated in the ChanParams.par
//#define PUMP_PMCA  PMCA_Jafri_Rice_Winslow_1998

#define RECEPTOR_AMPA AMPAR_DESTEXHE_MAINEN_SEJNOWSKI_1994
#define RECEPTOR_NMDA NMDAR_JAHR_STEVENS_1990 
#define RECEPTOR_GABAA GABAAR_DESTEXHE_MAINEN_SEJNOWSKI_1994

#define CHANNEL_RYR RYR2_WILLIAMS_JAFRI_2011
//#define CHANNEL_IP3R  IP3R_ULLAH_MAK_PEARSON_2012
#define CHANNEL_IP3R  IP3R_LI_RINZEL_1994
#define PUMP_SERCA  SERCA_Tran_Crampin_2009
//}}}

//#}}}
#elif MODEL_TO_USE == _MODEL_TESTING
//#include "ModelTesting.h"
#define SIMULATE_VM
#define SIMULATE_CACYTO
#define ADAPTIVE_IO
//{{{
  #define CHANNEL_NAT NAT_HODGKIN_HUXLEY_1952 
  #define CHANNEL_KDR KDR_HODGKIN_HUXLEY_1952 
//}}}
#endif

//////////////////////////////////////////////////////////////////////
// Default setting
//
//{{{
#ifndef CHANNEL_NAT
  #define CHANNEL_NAT _NAT_DEFAULT
#endif
#ifndef CHANNEL_NAT_AIS
  #define CHANNEL_NAT_AIS _NAT_AIS_DEFAULT
#endif
#ifndef CHANNEL_NAP
#define CHANNEL_NAP _NAP_DEFAULT
#endif
#ifndef CHANNEL_NAS
#define CHANNEL_NAS _NAS_DEFAULT
#endif
#ifndef CHANNEL_KAf
#define CHANNEL_KAf _KAf_DEFAULT
#endif
#ifndef CHANNEL_KAs
#define CHANNEL_KAs _KAs_DEFAULT
#endif
#ifndef CHANNEL_KIR
#define CHANNEL_KIR _KIR_DEFAULT
#endif
#ifndef CHANNEL_KRP
#define CHANNEL_KRP _KRP_DEFAULT
#endif
#ifndef CHANNEL_BK
#define CHANNEL_BK _BK_DEFAULT
#endif
#ifndef CHANNEL_BKalpha
#define CHANNEL_BKalpha _BKalphabeta_DEFAULT
#endif
#ifndef CHANNEL_BKalphabeta
#define CHANNEL_BKalphabeta _BKalphabeta_DEFAULT
#endif
#ifndef CHANNEL_SK
#define CHANNEL_SK _SK_DEFAULT
#endif
#ifndef CHANNEL_KDR
#define CHANNEL_KDR _KDR_DEFAULT
#endif
#ifndef CHANNEL_KDR_AIS
#define CHANNEL_KDR_AIS _KDR_AIS_DEFAULT
#endif
#ifndef CHANNEL_MK
#define CHANNEL_MK _MK_DEFAULT
#endif
#ifndef CHANNEL_Kv31
#define CHANNEL_Kv31 _Kv31_DEFAULT
#endif
#ifndef CHANNEL_CaHVA
#define CHANNEL_CaHVA _CaHVA_DEFAULT
#endif
#ifndef CHANNEL_CaLVA
#define CHANNEL_CaLVA _CaLVA_DEFAULT
#endif
#ifndef CHANNEL_CaLv12
#define CHANNEL_CaLv12 _CaLv12_DEFAULT
#endif
#ifndef CHANNEL_CaLv13
#define CHANNEL_CaLv13 _CaLv13_DEFAULT
#endif
#ifndef CHANNEL_CaN
#define CHANNEL_CaN _CaN_DEFAULT
#endif
#ifndef CHANNEL_CaPQ
#define CHANNEL_CaPQ _CaPQ_DEFAULT
#endif
#ifndef CHANNEL_CaR
#define CHANNEL_CaR _CaR_DEFAULT
#endif
#ifndef CHANNEL_CaT
#define CHANNEL_CaT _CaT_DEFAULT
#endif
#ifndef RECEPTOR_AMPA
#define RECEPTOR_AMPA _AMPAR_DEFAULT
#endif
#ifndef RECEPTOR_NMDA
#define RECEPTOR_NMDA _NMDAR_DEFAULT
#endif
#ifndef RECEPTOR_GABAA
#define RECEPTOR_GABAA _GABAAR_DEFAULT
#endif
#ifndef RECEPTOR_GABAB
#define RECEPTOR_GABAB _GABABR_DEFAULT
#endif

#ifndef PUMP_PMCA
#define PUMP_PMCA _PMCA_DEFAULT
#endif
#ifndef EXCHANGER_NCX
#define EXCHANGER_NCX _NCX_DEFAULT
#endif
#ifndef CHANNEL_RYR
#define CHANNEL_RYR _RYR_DEFAULT
#endif
#ifndef CHANNEL_IP3R
#define CHANNEL_IP3R _IP3R_DEFAULT
#endif
#ifndef PUMP_SERCA
#define PUMP_SERCA _SERCA_DEFAULT
#endif

#ifndef CHANNEL_HCN
  #define CHANNEL_HCN _HCN_DEFAULT
#endif

#ifndef MODEL_TO_USE
#define MODEL_TO_USE _MODEL_NOT_DEFINED
#endif
	 // implicit synapse space
#ifndef SYNAPSE_MODEL_STRATEGY
#define SYNAPSE_MODEL_STRATEGY USE_PRESYNAPTICPOINT
#endif

	 // if explicit synapse space is used
//#if SYNAPSE_MODEL_STRATEGY == USE_SYNAPTICCLEFT 
	 // default: use simple estimation of neurotransmitter as 
	 // given in Dextexhe-Mainen-Sejnowski 1994 
#ifndef GLUTAMATE_UPDATE_METHOD
#define GLUTAMATE_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#endif
#ifndef GABA_UPDATE_METHOD
#define GABA_UPDATE_METHOD NEUROTRANSMITTER_DESTEXHE_MAINEN_SEJNOWSKI_1994
#endif
//#endif

//#ifndef SIMULATION_INVOLVE
//#define SIMULATION_INVOLVE  VMONLY
//#endif
#ifndef CALCIUM_CYTO_DYNAMICS
  #define CALCIUM_CYTO_DYNAMICS FAST_BUFFERING
#endif
#ifndef CALCIUM_ER_DYNAMICS
  #define CALCIUM_ER_DYNAMICS FAST_BUFFERING
#endif
#ifndef IP3_CYTO_DYNAMICS
  #define IP3_CYTO_DYNAMICS FAST_BUFFERING
#endif

//}}}


#endif
