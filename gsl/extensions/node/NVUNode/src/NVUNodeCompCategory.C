#include "Lens.h"
#include "NVUNodeCompCategory.h"
#include "NDPairList.h"
#include "CG_NVUNodeCompCategory.h"

NVUNodeCompCategory::NVUNodeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_NVUNodeCompCategory(sim, modelName, ndpList)
{
}

void NVUNodeCompCategory::paramInitalize(RNG& rng) 
{
    // general constants:
    getSharedMembers().Farad       = 96500         ;// [C mol-1] Faradays constant.
    getSharedMembers().R_gas       = 8.315         ;// [J mol-1K-1]
    getSharedMembers().Temp        = 300           ;// [K]
    getSharedMembers().unitcon     = 1e3           ;// [-] Factor to convert equations to another unit.

// NE & AC constants:
    getSharedMembers().L_p         = 2.1e-9        ;// [m uM-1s-1]
    getSharedMembers().R_tot       = 8.79e-8       ;// [m]   total volume surface area ratio AC+SC
    getSharedMembers().X_k         = 12.41e-3      ;// [uMm]
    getSharedMembers().z_Na        = 1             ;// [-]
    getSharedMembers().z_K         = 1             ;// [-]
    getSharedMembers().z_Cl        = -1            ;// [-]
    getSharedMembers().z_NBC       = -1            ;// [-]
    getSharedMembers().g_K_k       = 40            ;// [ohm-1m-2]
    getSharedMembers().g_KCC1_k    = 1e-2          ;// [ohm-1m-2]
    getSharedMembers().g_NBC_k     = 7.57e-1       ;// [ohm-1m-2]
    getSharedMembers().g_Cl_k      = 8.797e-1      ;// [ohm-1m-2]
    getSharedMembers().g_NKCC1_k   = 5.54e-2       ;// [ohm-1m-2]
    getSharedMembers().g_Na_k      = 1.314         ;// [ohm-1m-2]
    getSharedMembers().J_NaK_max   = 1.42e-3       ;// [uMm s-1]
    getSharedMembers().K_Na_k      = 10e3          ;// [uM]
    getSharedMembers().K_K_s       = 1.5e3         ;// [uM]
    getSharedMembers().k_C         = 7.35e-5       ;// [muM s-1]

// Perivascular Space constants:
    getSharedMembers().R_decay     = 0.05;  // s^-1
    getSharedMembers().K_p_min   = 3e3;     // uM

// BK channel constants:
    getSharedMembers().A_ef_k      = 3.7e-9        ;                // m2       Area of an endfoot of an astrocyte, equal to Area astrocyte at synaptic cleft
    getSharedMembers().v_4         = 8e-3       ;               // V        A measure of the spread of the distribution
    getSharedMembers().psi_w       = 2.664         ;                // s-1      A characteristic time
    getSharedMembers().G_BK_k      = 225         ;              // !!!
    getSharedMembers().g_BK_k      = getSharedMembers().G_BK_k * 1e-12 / getSharedMembers().A_ef_k ;    // ohm-1m-2  Specific capacitance of the BK-Channel in units of Ostby
    getSharedMembers().VR_pa       = 0.001           ;              // [-]       The estimated volume ratio of perivascular space to astrocyte: Model estimation
    getSharedMembers().VR_ps       = 0.001         ;                // [-]       The estimated volume ratio of perivascular space to SMC: Model Estimation

// SMC constants:
    getSharedMembers().F_il         = 7.5e2;        //[-] scaling factor to fit the experimental data of Filosa
    getSharedMembers().z_1      =4.5;           //[-] parameter fitted on experimental data of Filosa
    getSharedMembers().z_2      =-1.12e2;       //[-] parameter fitted on experimental data of Filosa
    getSharedMembers().z_3      =4.2e-1;        //[-] parameter fitted on experimental data of Filosa
    getSharedMembers().z_4      =-1.26e1;       //[-] parameter fitted on experimental data of Filosa
    getSharedMembers().z_5      =-7.4e-2;       //[-] parameter fitted on experimental data of Filosa
    getSharedMembers().Fmax_i       = 0.23;         // (microM/s)
    getSharedMembers().Kr_i         = 1;            // (microM) Half saturation constant for agonist-dependent Ca entry
    getSharedMembers().G_Ca     = 0.00129;      // (microM/mV/s)
    getSharedMembers().v_Ca1        = 100;          // (mV)
    getSharedMembers().v_Ca2        = -24;          // (mV)
    getSharedMembers().R_Ca     = 8.5;          // (mV)
    getSharedMembers().G_NaCa       = 0.00316;      // (microM/mV/s)
    getSharedMembers().c_NaCa       = 0.5;          // (microM)
    getSharedMembers().v_NaCa       = -30;
    getSharedMembers().B_i      = 2.025;
    getSharedMembers().cb_i     = 1;
    getSharedMembers().CICR_rate      = 55;
    getSharedMembers().sc_i     = 2;
    getSharedMembers().cc_i     = 0.9;
    getSharedMembers().D_i      = 0.24;
    getSharedMembers().vd_i     = -100;
    getSharedMembers().Rd_i     = 250;
    getSharedMembers().L_i      = 0.025;
    getSharedMembers().delta_mv      = 1970;         // mVmicroM-1 The change in membrane potential by a scaling factor
    getSharedMembers().F_NaK        = 0.0432;
    getSharedMembers().G_Cl     = 0.00134;
    getSharedMembers().v_Cl     = -25;
    getSharedMembers().G_K      = 0.00446;
    getSharedMembers().vK_i     = -94;
    getSharedMembers().lam      = 45;
    getSharedMembers().v_Ca3        = -27;          // correct
    getSharedMembers().R_K      = 12;
    getSharedMembers().const_k_i      = 0.1;

// Stretch-activated channels
    getSharedMembers().G_stretch   = 0.0061;       // uM mV-1 s-1   (stretch activated channels)
    getSharedMembers().Esac        = -18;          // mV
    getSharedMembers().alpha1      = 0.0074;
    getSharedMembers().sig0        = 500;

// EC constants:
    getSharedMembers().Fmax_j       = 0.23;         // [microM/s]
    getSharedMembers().Kr_j     = 1;
    getSharedMembers().B_j      = 0.5;
    getSharedMembers().cb_j     = 1;
    getSharedMembers().C_j      = 5;
    getSharedMembers().sc_j     = 2;
    getSharedMembers().cc_j     = 0.9;
    getSharedMembers().D_j      = 0.24;
    getSharedMembers().L_j      = 0.025;
    getSharedMembers().G_cat        = 0.66e-3;      //!
    getSharedMembers().E_Ca     = 50;
    getSharedMembers().m3cat        = -0.18;        //-6.18 changed value!
    getSharedMembers().m4cat        = 0.37;
    getSharedMembers().JO_j         = 0.029;        //constant Ca influx (EC)
    getSharedMembers().C_m      = 25.8;
    getSharedMembers().G_tot        = 6927;
    getSharedMembers().vK_j         = -80;
    getSharedMembers().const_a1           = 53.3;
    getSharedMembers().a2           = 53.3;
    getSharedMembers().const_b            = -80.8;
    getSharedMembers().const_c            = -0.4;         //-6.4 changed value!
    getSharedMembers().m3b      = 1.32e-3;
    getSharedMembers().m4b      = 0.3;
    getSharedMembers().m3s      = -0.28;
    getSharedMembers().m4s      = 0.389;
    getSharedMembers().G_R      = 955;
    getSharedMembers().v_rest       = -31.1;
    getSharedMembers().const_k_j      = 0.1;
    getSharedMembers().J_PLC        = 0.11; //0.11 or 0.3 *****************************
    getSharedMembers().g_hat      = 0.5;
    getSharedMembers().p_hat      = 0.05;
    getSharedMembers().p_hatIP3   = 0.05;
    getSharedMembers().C_Hillmann = 1;
    getSharedMembers().K3_c        = 0.4 * getSharedMembers().C_Hillmann;
    getSharedMembers().K4_c        = 0.1 * getSharedMembers().C_Hillmann;
    getSharedMembers().K7_c        = 0.1 * getSharedMembers().C_Hillmann;
    getSharedMembers().gam_cross   = 17 * getSharedMembers().C_Hillmann;
    getSharedMembers().LArg_j        = 100;

    // ECS:
    // tau is dx^2 / 2D where dx is the length and D is diffusion rate
    getSharedMembers().tau2          = 2.8;     // (sec) characteristic time scale for ion to travel from PVS to SC (AC length is ~100 um, based on protoplasmic astrocyte process length of ~50 um)

    // NO pathway

    getSharedMembers().LArg        = 100;
    getSharedMembers().V_spine     = 8e-8;
    getSharedMembers().k_ex        = 1600;
    getSharedMembers().Ca_rest     = 0.1;
    getSharedMembers().lambda      = 20;
    getSharedMembers().V_maxNOS    = 25e-3;
    getSharedMembers().V_max_NO_n  = 4.22;
    getSharedMembers().K_mO2_n   = 243;
    getSharedMembers().K_mArg_n  = 1.5;
    getSharedMembers().K_actNOS    = 9.27e-2;
    getSharedMembers().D_NO          = 3300;
    getSharedMembers().k_O2        = 9.6e-6;
    getSharedMembers().const_On          = 200;
    getSharedMembers().v_n         = -0.04;
    getSharedMembers().const_Ok          = 200;
    getSharedMembers().G_M         = 46000;
    getSharedMembers().dist_nk     = 25;
    getSharedMembers().dist_ki     = 25;
    getSharedMembers().dist_ij     = 3.75;
    getSharedMembers().tau_nk      = pow(getSharedMembers().dist_nk,2)/(2*getSharedMembers().D_NO);
    getSharedMembers().tau_ki      = pow(getSharedMembers().dist_ki,2)/(2*getSharedMembers().D_NO);
    getSharedMembers().tau_ij      = pow(getSharedMembers().dist_ij,2)/(2*getSharedMembers().D_NO);
    getSharedMembers().P_Ca_P_M    = 3.6;
    getSharedMembers().Ca_ex       = 2e3;
    getSharedMembers().const_M           = 1.3e5;
    getSharedMembers().betA        = 650 ;
    getSharedMembers().betB        = 2800 ;
    getSharedMembers().const_Oj          = 200;
    getSharedMembers().K_dis       = 9e-2;
    getSharedMembers().K_eNOS      = 4.5e-1;
    getSharedMembers().mu2         = 0.0167;
    getSharedMembers().g_max       = 0.06;
    getSharedMembers().const_alp         = 2;
    getSharedMembers().W_0         = 1.4;
    getSharedMembers().delt_wss    = 2.86;
    getSharedMembers().k_dno       = 0.01;
    getSharedMembers().k1          = 2e3 ;
    getSharedMembers().k2          = 0.1;
    getSharedMembers().k3          = 3;
    getSharedMembers().k_1         = 100;
    getSharedMembers().V_max_sGC   = 0.8520;  //\muM s{-1}; (for m = 2)
    getSharedMembers().k_pde       = 0.0195;// s{-1} (for m = 2)
    getSharedMembers().C_4         = 0.011; // [s{-1} microM{-2}] (note: the changing units are correct!) (for m = 2)
    getSharedMembers().K_m_pde     = 2;                 // [microM]
    getSharedMembers().k_mlcp_b    = 0.0086;         // [s{-1}]
    getSharedMembers().k_mlcp_c    = 0.0327;          //[s{-1}]
    getSharedMembers().K_m_mlcp    = 5.5;               // [microM]
    getSharedMembers().bet_i       = 0.13; // translation factor for membrane potential dependence of KCa channel activation sigmoidal [microM2]
    getSharedMembers().const_m             = 2;
    getSharedMembers().gam_eNOS    = 0.1; // [-]
    getSharedMembers().K_mO2_j     = 7.7;
    getSharedMembers().V_NOj_max   = 1.22;
    getSharedMembers().K_mArg_j    = 1.5;

    // AC Ca2+
    getSharedMembers().r_buff           = 0.05;
    getSharedMembers().G_TRPV_k     = 50;
    getSharedMembers().g_TRPV_k     = getSharedMembers().G_TRPV_k * 1e-12 / getSharedMembers().A_ef_k;
    getSharedMembers().J_max            = 2880;
    getSharedMembers().const_K_act            = 0.17;
    getSharedMembers().K_I          = 0.03;
    getSharedMembers().P_L          = 0.0804;
    getSharedMembers().k_pump           = 0.24;
    getSharedMembers().const_V_max            = 20;
    getSharedMembers().C_astr_k     = 40;
    getSharedMembers().gamma_k      = 834.3;
    getSharedMembers().B_ex             = 11.35;
    getSharedMembers().BK_end           = 40;
    getSharedMembers().K_ex         = 0.26;
    getSharedMembers().const_delta            = 1.235e-2;
    getSharedMembers().K_G          = 8.82;
    getSharedMembers().Ca_3         = 0.4;
    getSharedMembers().Ca_4         = 0.35;
    getSharedMembers().v_5          = 15e-3;
    getSharedMembers().v_7          = -55e-3;
    getSharedMembers().eet_shift        = 2e-3;
    getSharedMembers().gam_cae_k        = 200;
    getSharedMembers().gam_cai_k        = 0.01;
    getSharedMembers().R_0_passive_k    = 20e-6;
    getSharedMembers().epshalf_k        = 0.1;
    getSharedMembers().kappa_k      = 0.1;
    getSharedMembers().v1_TRPV_k        = 0.12;
    getSharedMembers().v2_TRPV_k        = 0.013;
    getSharedMembers().t_TRPV_k     = 0.9;
    getSharedMembers().VR_ER_cyt        = 0.185;
    getSharedMembers().K_inh            = 0.1;
    getSharedMembers().k_on         = 2;
    getSharedMembers().k_deg            = 1.25;
    getSharedMembers().r_h          = 4.8;
    getSharedMembers().Ca_k_min     = 0.1;
    getSharedMembers().k_eet            = 7.2;
    getSharedMembers().V_eet            = 72;
    getSharedMembers().Ca_decay_k       = 0.5;
    getSharedMembers().Capmin_k     = 2000;
    getSharedMembers().reverseBK        = 0;
    getSharedMembers().switchBK     = 1;
    getSharedMembers().trpv_switch  = 1;
    getSharedMembers().z_Ca         = 2;
    getSharedMembers().const_m_c          = 4;

    getSharedMembers().R0    = 10e-6 ;  // m (for nondimensionalising)
    getSharedMembers().P0    = 8000  ;  // Pa (scaling factor for nondim)
    getSharedMembers().PCAP  = 4000  ;  // Pa (capillary bed pressure)

    // Pressure constants
    getSharedMembers().HRR        = 0.1   ;  // Nondimensional (thickness to radius ratio)
    getSharedMembers().RSCALE     = 0.6   ;  // Dimensionless
    getSharedMembers().E0         = 66e3  ;  // Pa
    getSharedMembers().EPASSIVE   = 66e3  ;  // Pa
    getSharedMembers().EACTIVE    = 233e3 ;  // Pa
    getSharedMembers().ETA        = 1e4;//2.8e2 ;  // Pa s
    getSharedMembers().T0         = 1     ;  // s


    getSharedMembers().PA2MMHG   = 0.00750061683; // convert from Pa to mmHg
    getSharedMembers().tau_diffusion   = 4.3; // for ecs K+
}
