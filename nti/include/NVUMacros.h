// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef _NVUMacros_H
#define _NVUMacros_H
/*
@ University of Canterbury 2017-2018. All rights reserved.
*/

// NVU
//{{{
#define NEQ  42   // those solved by ODE
#define NEQ_EXTRA 2   // those NVU receives (from NTS?) but want to export along with it
#define NEQ_TOTAL (NEQ + NEQ_EXTRA) // those Htree export

#define DIMENSIONS 3  // 3-dimensional space

//#define NTS_INTEGRATION	0  // to be removed


// Here keep track the indices of the associated variable name in 
//   1. stateVar array
//   2. flux     array

// total(stateVar) = 
// 6+3+1+6+4+3+1+10+7 = 41
//{{{
#define i_radius   0 // radius has to be 0, this is assumed elsewhere
// AC  6
//{{{ 
#define R_k        1
#define N_Na_k     2
#define N_K_k      3
#define N_HCO3_k   4
#define N_Cl_k     5
#define w_k        10
//}}}

// SC  3
//{{{ 
#define N_Na_s     6
#define N_K_s      7
#define N_HCO3_s   8
//}}}

// PVS  1
//{{{ 
#define K_p        9
//}}}

// SMC  6
//{{{ 
#define ca_i       11
#define ca_sr_i    12
#define v_i        13
#define w_i        14
#define ip3_i      15
#define K_i        16
//}}}

// EC  4
//{{{ 
#define ca_j       17
#define ca_er_j    18
#define v_j        19
#define ip3_j      20
//}}}

// Mech  3
//{{{ 
#define Mp         21
#define AMp        22
#define AM         23
//}}}

// ECS  1
//{{{ 
#define K_e      24
//}}}

// NO pathway  10
//{{{ 
#define NOn         25
#define NOk         26
#define NOi         27
#define NOj         28
#define cGMP        29
#define eNOS        30
#define nNOS        31
#define ca_n        32
#define E_b         33
#define E_6c        34
//}}}

// AC Ca2+   7
//{{{ 
#define ca_k        35
#define s_k         36
#define h_k         37
#define ip3_k       38
#define eet_k       39
#define m_k         40
#define ca_p        41
//}}}
//}}}

// NVU fluxes  103
//{{{ 
#define flu_pt  0
#define flu_P_str 1
#define flu_delta_p 2
#define flu_R_s 3
#define flu_N_Cl_s  4
#define flu_Na_s  5
#define flu_K_s 6
#define flu_HCO3_s  7
#define flu_Cl_s  8
#define flu_E_TRPV_k  9
#define flu_Na_k  10
#define flu_K_k 11
#define flu_HCO3_k  12
#define flu_Cl_k  13
#define flu_E_Na_k  14
#define flu_E_K_k 15
#define flu_E_Cl_k  16
#define flu_E_NBC_k 17
#define flu_E_BK_k  18
#define flu_J_NaK_k 19
#define flu_v_k 20
#define flu_J_KCC1_k  21
#define flu_J_NBC_k 22
#define flu_J_NKCC1_k 23
#define flu_J_Na_k  24
#define flu_J_K_k 25
#define flu_J_BK_k  26
#define flu_M 27
#define flu_h_r 28
#define flu_v_cpl_i 29
#define flu_c_cpl_i 30
#define flu_I_cpl_i 31
#define flu_rho_i 32
#define flu_ip3_i 33
#define flu_SRuptake_i  34
#define flu_CICR_i  35
#define flu_extrusion_i 36
#define flu_leak_i  37
#define flu_VOCC_i  38
#define flu_NaCa_i  39
#define flu_NaK_i 40
#define flu_Cl_i  41
#define flu_K_i 42
#define flu_degrad_i  43
#define flu_J_stretch_i 44
#define flu_v_KIR_i 45
#define flu_G_KIR_i 46
#define flu_J_KIR_i 47
#define flu_v_cpl_j 48
#define flu_c_cpl_j 49
#define flu_I_cpl_j 50
#define flu_rho_j 51
#define flu_O_j 52
#define flu_ip3_j 53
#define flu_ERuptake_j  54
#define flu_CICR_j  55
#define flu_extrusion_j 56
#define flu_leak_j  57
#define flu_cation_j  58
#define flu_BKCa_j  59
#define flu_SKCa_j  60
#define flu_K_j 61
#define flu_R_j 62
#define flu_degrad_j  63
#define flu_J_stretch_j 64
#define flu_K1_c  65
#define flu_K6_c  66
#define flu_F_r 67
#define flu_E 68
#define flu_R_0 69
#define flu_P_NR2AO 70
#define flu_P_NR2BO 71
#define flu_I_Ca  72
#define flu_CaM 73
#define flu_tau_w 74
#define flu_W_tau_w 75
#define flu_F_tau_w 76
#define flu_k4  77
#define flu_R_cGMP2 78
#define flu_K2_c  79
#define flu_K5_c  80
#define flu_c_w 81
#define flu_Kactivation_i 82
#define flu_E_5c  83
#define flu_V_max_pde 84
#define flu_rho 85
#define flu_ip3_k 86
#define flu_er_leak 87
#define flu_pump_k  88
#define flu_I_TRPV_k  89
#define flu_TRPV_k  90
#define flu_B_cyt 91
#define flu_G 92
#define flu_v_3 93
#define flu_w_inf 94
#define flu_phi_w 95
#define flu_H_Ca_k  96
#define flu_eta 97
#define flu_minf_k  98
#define flu_t_Ca_k  99
#define flu_VOCC_k  100
#define flu_K_input 101
#define flu_nvu_Glu 102
//}}}

//}}}


#endif
