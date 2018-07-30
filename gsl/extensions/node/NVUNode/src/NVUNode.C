#include "Lens.h"
#include "NVUNode.h"
#include "CG_NVUNode.h"
#include "rndm.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "Coordinates.h"
#include <algorithm>

/****** Model parameters ******/

// Pressure constants

#if 1   //group large code
#define L0 (getSharedMembers().L0)

#define PA2MMHG (getSharedMembers().PA2MMHG)
#define T0 (getSharedMembers().T0)
#define ETA (getSharedMembers().ETA)
#define EACTIVE (getSharedMembers().EACTIVE)
#define EPASSIVE (getSharedMembers().EPASSIVE)
#define E0 (getSharedMembers().E0)
#define RSCALE (getSharedMembers().RSCALE)
#define HRR (getSharedMembers().HRR)
#define R0 (getSharedMembers().R0)
#define P0 (getSharedMembers().P0)
#define PCAP (getSharedMembers().PCAP)

// general constants:
#define Farad (getSharedMembers().Farad)
#define R_gas (getSharedMembers().R_gas)
#define Temp (getSharedMembers().Temp)
#define unitcon (getSharedMembers().unitcon)

// NE & AC constants:
#define L_p (getSharedMembers().L_p)
#define R_tot (getSharedMembers().R_tot)
#define X_k (getSharedMembers().X_k)
#define z_Na (getSharedMembers().z_Na)
#define z_K (getSharedMembers().z_K)
#define z_Cl (getSharedMembers().z_Cl)
#define z_NBC (getSharedMembers().z_NBC)
#define g_K_k (getSharedMembers().g_K_k)
#define g_KCC1_k (getSharedMembers().g_KCC1_k)
#define g_NBC_k (getSharedMembers().g_NBC_k)
#define g_Cl_k (getSharedMembers().g_Cl_k)
#define g_NKCC1_k (getSharedMembers().g_NKCC1_k)
#define g_Na_k (getSharedMembers().g_Na_k)
#define J_NaK_max (getSharedMembers().J_NaK_max)
#define K_Na_k (getSharedMembers().K_Na_k)
#define K_K_s (getSharedMembers().K_K_s)
#define k_C (getSharedMembers().k_C)

// Perivascular Space constants:
#define R_decay (getSharedMembers().R_decay)
#define K_p_min (getSharedMembers().K_p_min)

// BK channel constants:
#define A_ef_k (getSharedMembers().A_ef_k)
#define v_4 (getSharedMembers().v_4)
#define psi_w (getSharedMembers().psi_w)
#define G_BK_k (getSharedMembers().G_BK_k)
#define g_BK_k (getSharedMembers().g_BK_k)
#define VR_pa (getSharedMembers().VR_pa)
#define VR_ps (getSharedMembers().VR_ps)

// SMC constants:
#define F_il (getSharedMembers().F_il)
#define z_1 (getSharedMembers().z_1)
#define z_2 (getSharedMembers().z_2)
#define z_3 (getSharedMembers().z_3)
#define z_4 (getSharedMembers().z_4)
#define z_5 (getSharedMembers().z_5)
#define Fmax_i (getSharedMembers().Fmax_i)
#define Kr_i (getSharedMembers().Kr_i)
#define G_Ca (getSharedMembers().G_Ca)
#define v_Ca1 (getSharedMembers().v_Ca1)
#define v_Ca2 (getSharedMembers().v_Ca2)
#define R_Ca (getSharedMembers().R_Ca)
#define G_NaCa (getSharedMembers().G_NaCa)
#define c_NaCa (getSharedMembers().c_NaCa)
#define v_NaCa (getSharedMembers().v_NaCa)
#define B_i (getSharedMembers().B_i)
#define cb_i (getSharedMembers().cb_i)
#define CICR_rate (getSharedMembers().CICR_rate) // FIXed from C_i
#define sc_i (getSharedMembers().sc_i)
#define cc_i (getSharedMembers().cc_i)
#define D_i (getSharedMembers().D_i)
#define vd_i (getSharedMembers().vd_i)
#define Rd_i (getSharedMembers().Rd_i)
#define L_i (getSharedMembers().L_i)
#define delta_mv (getSharedMembers().delta_mv) // FIXED from gam
#define F_NaK (getSharedMembers().F_NaK)
#define G_Cl (getSharedMembers().G_Cl)
#define v_Cl (getSharedMembers().v_Cl)
#define G_K (getSharedMembers().G_K)
#define vK_i (getSharedMembers().vK_i)
#define lam (getSharedMembers().lam)
#define v_Ca3 (getSharedMembers().v_Ca3)
#define R_K (getSharedMembers().R_K)
#define const_k_i (getSharedMembers().const_k_i) //FIXed from k_i

// Stretch-activated channels
#define G_stretch (getSharedMembers().G_stretch)
#define Esac (getSharedMembers().Esac)
#define alpha1 (getSharedMembers().alpha1)
#define sig0 (getSharedMembers().sig0)

// EC constants:
#define Fmax_j (getSharedMembers().Fmax_j)
#define Kr_j (getSharedMembers().Kr_j)
#define B_j (getSharedMembers().B_j)
#define cb_j (getSharedMembers().cb_j)
#define C_j (getSharedMembers().C_j)
#define sc_j (getSharedMembers().sc_j)
#define cc_j (getSharedMembers().cc_j)
#define D_j (getSharedMembers().D_j)
#define L_j (getSharedMembers().L_j)
#define G_cat (getSharedMembers().G_cat)
#define E_Ca (getSharedMembers().E_Ca)
#define m3cat (getSharedMembers().m3cat)
#define m4cat (getSharedMembers().m4cat)
#define JO_j (getSharedMembers().JO_j)
#define C_m (getSharedMembers().C_m)
#define G_tot (getSharedMembers().G_tot)
#define vK_j (getSharedMembers().vK_j)
#define const_a1 (getSharedMembers().const_a1) // FIXed from a1
#define a2 (getSharedMembers().a2)
#define const_b (getSharedMembers().const_b) // FIXed from b
#define const_c (getSharedMembers().const_c) // FIXed from c
#define m3b (getSharedMembers().m3b)
#define m4b (getSharedMembers().m4b)
#define m3s (getSharedMembers().m3s)
#define m4s (getSharedMembers().m4s)
#define G_R (getSharedMembers().G_R)
#define v_rest (getSharedMembers().v_rest)
#define const_k_j (getSharedMembers().const_k_j) // FIXed from k_j
#define J_PLC (getSharedMembers().J_PLC)
#define g_hat (getSharedMembers().g_hat)
#define p_hat (getSharedMembers().p_hat)
#define p_hatIP3 (getSharedMembers().p_hatIP3)
#define C_Hillmann (getSharedMembers().C_Hillmann)
#define K3_c (getSharedMembers().K3_c)
#define K4_c (getSharedMembers().K4_c)
#define K7_c (getSharedMembers().K7_c)
#define gam_cross (getSharedMembers().gam_cross)
#define LArg_j (getSharedMembers().LArg_j)

// ECS:
// tau is dx^2 / 2D where dx is the length and D is diffusion rate
#define tau2 (getSharedMembers().tau2)

// NO pathway
#define LArg (getSharedMembers().LArg)
#define V_spine (getSharedMembers().V_spine)
#define k_ex (getSharedMembers().k_ex)
#define Ca_rest (getSharedMembers().Ca_rest)
#define lambda (getSharedMembers().lambda)
#define V_maxNOS (getSharedMembers().V_maxNOS)
#define V_max_NO_n (getSharedMembers().V_max_NO_n)
#define K_mO2_n (getSharedMembers().K_mO2_n)
#define K_mArg_n (getSharedMembers().K_mArg_n)
#define K_actNOS (getSharedMembers().K_actNOS)
#define D_NO (getSharedMembers().D_NO)
#define k_O2 (getSharedMembers().k_O2)
#define const_On (getSharedMembers().const_On) // FIXed fom On
#define v_n (getSharedMembers().v_n)
#define const_Ok (getSharedMembers().const_Ok) //FIXed from Ok
#define G_M (getSharedMembers().G_M)
#define dist_nk (getSharedMembers().dist_nk)
#define dist_ki (getSharedMembers().dist_ki)
#define dist_ij (getSharedMembers().dist_ij)
#define tau_nk (getSharedMembers().tau_nk)
#define tau_ki (getSharedMembers().tau_ki)
#define tau_ij (getSharedMembers().tau_ij)
#define P_Ca_P_M (getSharedMembers().P_Ca_P_M)
#define Ca_ex (getSharedMembers().Ca_ex)
#define const_M (getSharedMembers().const_M) // FIXed from M
#define betA (getSharedMembers().betA)
#define betB (getSharedMembers().betB)
#define const_Oj (getSharedMembers().const_Oj) //FIXed from Oj
#define K_dis (getSharedMembers().K_dis)
#define K_eNOS (getSharedMembers().K_eNOS)
#define mu2 (getSharedMembers().mu2)
#define g_max (getSharedMembers().g_max)
#define const_alp (getSharedMembers().const_alp) // FIXed from alp
#define W_0 (getSharedMembers().W_0)
#define delt_wss (getSharedMembers().delt_wss)
#define k_dno (getSharedMembers().k_dno)
#define k1 (getSharedMembers().k1)
#define k2 (getSharedMembers().k2)
#define k3 (getSharedMembers().k3)
#define k_1 (getSharedMembers().k_1)
#define V_max_sGC (getSharedMembers().V_max_sGC)
#define k_pde (getSharedMembers().k_pde)
#define C_4 (getSharedMembers().C_4)
#define K_m_pde (getSharedMembers().K_m_pde)
#define k_mlcp_b (getSharedMembers().k_mlcp_b)
#define k_mlcp_c (getSharedMembers().k_mlcp_c)
#define K_m_mlcp (getSharedMembers().K_m_mlcp)
#define bet_i (getSharedMembers().bet_i)
#define const_m (getSharedMembers().const_m) // FIXed from m
#define gam_eNOS (getSharedMembers().gam_eNOS)
#define K_mO2_j (getSharedMembers().K_mO2_j)
#define V_NOj_max (getSharedMembers().V_NOj_max)
#define K_mArg_j (getSharedMembers().K_mArg_j)

// AC Ca2+
#define r_buff (getSharedMembers().r_buff)
#define G_TRPV_k (getSharedMembers().G_TRPV_k)
#define g_TRPV_k (getSharedMembers().g_TRPV_k)
#define J_max (getSharedMembers().J_max)
#define const_K_act (getSharedMembers().const_K_act) // FIXed from K_act
#define K_I (getSharedMembers().K_I)
#define P_L (getSharedMembers().P_L)
#define k_pump (getSharedMembers().k_pump)
#define const_V_max (getSharedMembers().const_V_max) // FIXed from V_max
#define C_astr_k (getSharedMembers().C_astr_k)
#define gamma_k (getSharedMembers().gamma_k)
#define B_ex (getSharedMembers().B_ex)
#define BK_end (getSharedMembers().BK_end)
#define K_ex (getSharedMembers().K_ex)
#define const_delta (getSharedMembers().const_delta) // FIXed from delta
#define K_G (getSharedMembers().K_G)
#define Ca_3 (getSharedMembers().Ca_3)
#define Ca_4 (getSharedMembers().Ca_4)
#define v_5 (getSharedMembers().v_5)
#define v_7 (getSharedMembers().v_7)
#define eet_shift (getSharedMembers().eet_shift)
#define gam_cae_k (getSharedMembers().gam_cae_k)
#define gam_cai_k (getSharedMembers().gam_cai_k)
#define R_0_passive_k (getSharedMembers().R_0_passive_k)
#define epshalf_k (getSharedMembers().epshalf_k)
#define kappa_k (getSharedMembers().kappa_k)
#define v1_TRPV_k (getSharedMembers().v1_TRPV_k)
#define v2_TRPV_k (getSharedMembers().v2_TRPV_k)
#define t_TRPV_k (getSharedMembers().t_TRPV_k)
#define VR_ER_cyt (getSharedMembers().VR_ER_cyt)
#define K_inh (getSharedMembers().K_inh)
#define k_on (getSharedMembers().k_on)
#define k_deg (getSharedMembers().k_deg)
#define r_h (getSharedMembers().r_h)
#define Ca_k_min (getSharedMembers().Ca_k_min)
#define k_eet (getSharedMembers().k_eet)
#define V_eet (getSharedMembers().V_eet)
#define Ca_decay_k (getSharedMembers().Ca_decay_k)
#define Capmin_k (getSharedMembers().Capmin_k)
#define reverseBK (getSharedMembers().reverseBK)
#define switchBK (getSharedMembers().switchBK)
#define trpv_switch (getSharedMembers().trpv_switch)
#define z_Ca (getSharedMembers().z_Ca)
#define const_m_c (getSharedMembers().const_m_c) // FIXed from m_c

#define tau_diffusion (getSharedMembers().tau_diffusion) // for K+ ecs diff
#define TStep (*(getSharedMembers().deltaT) * 1e-3)  // sec
#define DEBUG
#define THRESHOLD_NUMSPIKES 2
#define GLUT_MIN 0.0  // uM
#define K_extra_baseline  3000.0 //uM
#define K_s_min  0.001  // uM
#endif

void NVUNode::initStateVariables(RNG& rng)
{
    if (not getSharedMembers().deltaT)
	std::cerr << "Please connect deltaT to NVU nodes";
    assert(getSharedMembers().deltaT);
#if INPUT_TO_NVU == OPTION_NEURON
    if (not numSpikes)
	std::cerr << "Please connect NTS (e.g. MegaSynapticSpace) to NVU nodes";
    assert(numSpikes);
#endif

    _gIdx =
	this->getGlobalIndex();
#if 1
    int rank = getSimulation().getRank();
    int comm_size = getSimulation().getNumProcesses();
    for (int i = 0; i < comm_size; i++)
    {
       MPI_Barrier(MPI_COMM_WORLD);
       std::vector<int> coords;
       int x, y;
       getNodeCoords(coords);
       if (i == rank and getSimulation().isSimulatePass())
       {
	  std::cerr << "AAA: " << rank << ": " 
	     << _gIdx << ", " << getIndex() 
	     << ", " << getNodeIndex() 
	     << ", " << coords[0] << "_" << coords[1] << "_" << coords[2] << "; "
	     << ", " << std::endl;
       }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    Glut_out = GLUT_MIN;
    //K_out = K_extra_baseline;
    J_K_out = 0.0;

    solver_init(); // calls nvu_ics within

    std::vector<int> indexCoordinate;
    std::vector<double> realCoordinate;

    // Set my own coordinates
    // Get index-based coordinates and
    // scale to physiological units based on NVUNode's length
    std::vector<int> gridSize =
	this->getGridLayerDescriptor()->getGrid()->getSize();
    getNodeCoords(indexCoordinate);
    //wrong:calculateRealCoordinates(indexCoordinate, getSharedMembers().L0, gridSize[0], gridSize[1], gridSize[2], realCoordinate);
    /* L0 in meter and thus coords are in meter */
    calculateRealCoordinatesNTS(indexCoordinate, L0,
	    gridSize[0], gridSize[1], gridSize[2], realCoordinate);
//#define DEBUG_2
#ifdef DEBUG_2
   std::cerr << "NVUnode with globalIdx = "
      << this->getGlobalIndex()
      << ", coord: " <<
      realCoordinate[0] << "," <<
      realCoordinate[1] << "," <<
      realCoordinate[2] << std::endl;
   std::cerr << std::endl;
#endif

    // Copy from vector to shallow array defined in mdl
    //coords.decreaseSizeTo(0);
    for (int i = 0; i < realCoordinate.size(); i++)
    {
        coords.push_back(realCoordinate[i]);
    }
#ifdef DEBUG
    printf("Node %d. x: %f, y: %f, z: %f\n",getNodeIndex(), coords[0], coords[1], coords[2]);
#endif

    //Set up the stateVariables array for the H tree
    for (int i = 0; i < NEQ; i++)
    {
    	stateVariables.push_back(workspace->y[i]);
    }
    /* add those from outside */
    add_or_update_extra_stateVariables(0); //0 = add, 1 = update (default)
    assert(stateVariables.size() == NEQ_TOTAL);

    // Map state variable in array to mdl level variable ECS K+ for diffusion to take place.
    K_ecs = workspace->y[K_e];

    state_r = workspace->y[i_radius];

    publicValue = value; // for LIFE

#if WRITE_STATE_VARS
    // Get the writer reader - open the state_var_file and write the column headings.
    if (getNodeIndex() == PLOTTING_NODE)    {
        printf("Writing State vars has been selected on node %d.\n",getNodeIndex());
    	workspace->state_var_file = fopen("NVUOutputStateVars.csv", "w");
    	fprintf(workspace->state_var_file, "#t, i_radius, R_k, N_Na_k, N_K_k, N_HCO3_k, N_Cl_k, w_k, N_Na_s, N_K_s, N_HCO3_s, K_p, ca_i, ca_sr_i, v_i, w_i, ip3_i, K_i, ca_j, ca_er_j, v_j, ip3_j, Mp, AMp, AM, K_e, NOn, NOk, NOi, NOj, cGMP, eNOS, nNOS, ca_n, E_b, E_6c, ca_k, s_k, h_k, ip3_k, eet_k, m_k, ca_p\n");
    }
#endif
#if WRITE_FLUXES
    if (getNodeIndex() == PLOTTING_NODE)    {
        printf("Writing fluxes has been selected on node %d.\n",getNodeIndex());
    	workspace->fluxes_file = fopen("NVUOutputFluxes.csv", "w");
    	fprintf(workspace->fluxes_file, "#t, flu_pt, flu_P_str, flu_delta_p, flu_R_s, flu_N_Cl_s, flu_Na_s, flu_K_s, flu_HCO3_s, flu_Cl_s, flu_E_TRPV_k, flu_Na_k, flu_K_k, flu_HCO3_k, flu_Cl_k, flu_E_Na_k, flu_E_K_k, flu_E_Cl_k, flu_E_NBC_k, flu_E_BK_k, flu_J_NaK_k, flu_v_k, flu_J_KCC1_k, flu_J_NBC_k, flu_J_NKCC1_k, flu_J_Na_k, flu_J_K_k, flu_J_BK_k, flu_M, flu_h_r, flu_v_cpl_i, flu_c_cpl_i, flu_I_cpl_i, flu_rho_i, flu_ip3_i, flu_SRuptake_i, flu_CICR_i, flu_extrusion_i, flu_leak_i, flu_VOCC_i, flu_NaCa_i, flu_NaK_i, flu_Cl_i, flu_K_i, flu_degrad_i, flu_J_stretch_i, flu_v_KIR_i, flu_G_KIR_i, flu_J_KIR_i, flu_v_cpl_j, flu_c_cpl_j, flu_I_cpl_j, flu_rho_j, flu_O_j, flu_ip3_j, flu_ERuptake_j, flu_CICR_j, flu_extrusion_j, flu_leak_j, flu_cation_j, flu_BKCa_j, flu_SKCa_j, flu_K_j, flu_R_j, flu_degrad_j, flu_J_stretch_j, flu_K1_c, flu_K6_c, flu_F_r, flu_E, flu_R_0, flu_P_NR2AO, flu_P_NR2BO, flu_I_Ca, flu_CaM, flu_tau_w, flu_W_tau_w, flu_F_tau_w, flu_k4, flu_R_cGMP2, flu_K2_c, flu_K5_c, flu_c_w, flu_Kactivation_i, flu_E_5c, flu_V_max_pde, flu_rho, flu_ip3_k, flu_er_leak, flu_pump_k, flu_I_TRPV_k, flu_TRPV_k, flu_B_cyt, flu_G, flu_v_3, flu_w_inf, flu_phi_w, flu_H_Ca_k, flu_eta, flu_minf_k, flu_t_Ca_k, flu_VOCC_k, flu_K_input, flu_nvu_Glu\n");
    }
#endif
}

void NVUNode::initJacobian(RNG& rng)
{
    calculate_pressure_index(); // Calculate my index in H tree coordinates array

    evaluate(workspace->t0, workspace->y, workspace->f);

    jacupdate(workspace->t0, workspace->y);

    workspace->S = newton_sparsity(workspace->J);

    newton_matrix();
    // from top of BE
    // These are 1D
    workspace->beta = zerosv(NEQ);
    workspace->w    = zerosv(NEQ); // TODO: do we really need two temp statevar arrays?
    workspace->x    = zerosv(NEQ);

    // Initial newton_matrix computation
    newton_matrix();

    workspace->jac_needed = 0; // flag to say that Jacobian is current
    workspace->converged = 0;
}

 // Fixed step Backward Euler ODE solver for ONE iteration
 // TODO: review this
 //
void NVUNode::update(RNG& rng)
{
#if INPUT_TO_NVU == OPTION_LIFE
    // LIFE
    workspace->counter++;
    if (workspace->counter >= LIFE_TIME * ITERATIONS_PER_SEC)
    {
        int neighborCount=0;
        ShallowArray<int*>::iterator iter, end = neighbors.end();
        for (iter=neighbors.begin(); iter!=end; ++iter) {
           neighborCount += **iter;
        }

        if (neighborCount<= getSharedMembers().tooSparse || neighborCount>=getSharedMembers().tooCrowded)
        {
            value=0;
        }
        else
        {
            workspace->injectStart = workspace->t;
            value=1;
        }
        workspace->counter = 0;
    }
#elif INPUT_TO_NVU == OPTION_NEURON
#ifdef DEBUG
    //if (LFP > -50)
    {
       //std::cerr << "nvu LFP: " << *voltage << " numSpikes = " << *numSpikes << std::endl;
       std::cerr << "nvu LFP: " << *voltage << " numSpikes = " << *numSpikes
	  << " time = " << workspace->t
	  << " Glut = " << Glut_out
	  << " K_sc(megaSC) = " << workspace->y[N_K_s]
	  << " K_o(EC) = " << workspace->y[K_e]
	  << " J_K_out = " << J_K_out
	  << " ECS_input= " << ECS_input(workspace->t)
	  << std::endl;
       //for (auto& n : Vm)
       //   std::cerr << " " <<  *n ;
    }
#endif
    nvu_Glu();
    nvu_K();
#endif

    // Perform a Jacobian update if necessary. This is in sync across
    if (workspace->jac_needed)
    {
        jacupdate(workspace->t, workspace->y);
        workspace->jac_needed = 0;
        newton_matrix();
    }

    // Copy values from previous completed timestep
    dcopy(NEQ, workspace->y, workspace->beta);
    dcopy(NEQ, workspace->y, workspace->w);
    workspace->tnext = workspace->t + TStep;

    // TODO: This code is stupid (straight from parbrain). What was parbrain trying to do instead??
    // There will never be more than a single loop... In parbrain, it doesn't do anything other than
    // act as an MPI barrier - waiting for all nodes to at least do one loop and flag they're done...

    // Indicate that we haven't converged yet
    workspace->converged = 0;

    // Newton loop
    for (int k = 0; k < workspace->maxits; k++)
    {
        evaluate(workspace->tnext, workspace->w, workspace->f); // f = g(w)
        // yeah not sure about this one.. original check was if every process has converged right? Should be fine...?
        if (workspace->converged)
        {
            if (k > workspace->nconv)
            {
                workspace->jac_needed = 1;
            }
            break; // w contains the correct value
        }

        // Form workspace->x = w - workspace->beta - dt g (our fcn value for Newton)
        dcopy(NEQ, workspace->w, workspace->x);
        daxpy(NEQ, -1, workspace->beta, workspace->x);
        daxpy(NEQ, -TStep, workspace->f, workspace->x);

        workspace->converged = 1;
    //TODO: this one is the actual implementation of the backward Euler
        lusoln(workspace->x);  // solve (workspace->x is now increment)
        daxpy(NEQ, -1, workspace->x, workspace->w); // update w with new value
    }

    if (!workspace->converged)
    {
        printf("Newton iteration failed to converge\n");
        exit(1);
    }
}


void NVUNode::copy(RNG& rng)
{
   workspace->t = workspace->tnext; //  is the same as getSimulation().getIteration() * TStep.
   dcopy(NEQ, workspace->w, workspace->y); // update y values

   // Copy new radius for H tree to see.
   state_r = workspace->y[i_radius];

   //Copy solution vector into statevariables array for H tree to see
   for (int i = 0; i < NEQ; i++)
   {
      stateVariables[i] = workspace->y[i];
   }
   add_or_update_extra_stateVariables(1); //0 = add, 1 = update (default)

   publicValue = value; // for LIFE
   K_ecs = workspace->y[K_e];


#if WRITE_STATE_VARS || WRITE_FLUXES
   if (fmod(workspace->t, workspace->dtwrite) < TStep && getNodeIndex() == PLOTTING_NODE)
   {
#if WRITE_STATE_VARS
      write_state_var_data();
#endif
#if WRITE_FLUXES
      write_fluxes_data();
#endif
   }
#endif

}

void NVUNode::finalize(RNG& rng)
{
    if (workspace->J != NULL) cs_spfree(workspace->J);
    if (workspace->dfdx->r != NULL) free(workspace->dfdx->r);
    if (workspace->dfdx->g != NULL) free(workspace->dfdx->g);
    if (workspace->dfdx->A != NULL) cs_spfree(workspace->dfdx->A);
    if (workspace->dfdx != NULL) free(workspace->dfdx);

    if (workspace->dfdx_pattern != NULL) cs_spfree(workspace->dfdx_pattern);

    if (workspace->N != NULL) cs_nfree(workspace->N);
    if (workspace->N != NULL) cs_di_sfree(workspace->S);
    if (workspace->y != NULL) free(workspace->y);
    if (workspace->f != NULL) free(workspace->f);
    if (workspace->fluxes != NULL) free(workspace->fluxes);
#if WRITE_STATE_VARS
    if (workspace->state_var_file != NULL) fclose(workspace->state_var_file);
#endif
#if WRITE_FLUXES
        if (workspace->fluxes_file != NULL) fclose(workspace->fluxes_file);
#endif
    if (workspace->beta != NULL) free(workspace->beta);
    if (workspace->x != NULL) free(workspace->x);
    if (workspace->w != NULL) free(workspace->w);
    if (workspace != NULL) free(workspace);
}



// TODO: These are the ICs used in parbrain, but they are not at steady state.
// When you are confident with your results, change them accodingly.
// NEQ  - here is NEQ=42 state variables
void NVUNode::nvu_ics()
{
   workspace->y[i_radius]  = 1.49986; // unit (maybe unitless)? - is this vessel radius?

   {//AC
   workspace->y[R_k]       = 6.20112e-8;
   workspace->y[N_Na_k]    = 0.00115629;
   workspace->y[N_K_k]     = 0.00554052;
   workspace->y[N_HCO3_k]  = 0.000582264;
   workspace->y[N_Cl_k]    = 0.000505576;
   workspace->y[w_k]       = 3.61562e-5;
   // AC Ca2+ pathway *******************
   workspace->y[ca_k]     = 0.133719;
   workspace->y[s_k]      = 502.461;
   workspace->y[h_k]      = 0.427865;
   workspace->y[ip3_k]    = 0.048299;
   workspace->y[eet_k]    = 0.337187;
   workspace->y[m_k]      = 0.896358;
   workspace->y[ca_p]     = 1713.39;
   }

   {//SC cleft 
//#define K_s_baseline 2811.28  // what unit ???
#define K_s   3639   // [uM] ~ 3.6 mM
#define VolumeArea_ratio_s (2e-8)   // [meter]
//#define K_s_baseline 7.27797e-5   // [uM.m]
#define K_s_baseline (K_s * VolumeArea_ratio_s)   // [uM.m]
// it is the concentration multiplied by the volume-area ratio of synaptic cleft 
// //TODO : ask Stewart to find out why K+ in synaptic cleft is too small?
#define Na_s   207135.5   // [uM] ~ 207.1 mM
#define Na_s_baseline (Na_s * VolumeArea_ratio_s)   // [uM.m]
#define HCO3_s   87667.24   // [uM] ~ 87.6 mM
#define HCO3_s_baseline (Na_s * VolumeArea_ratio_s)   // [uM.m]
      workspace->y[N_K_s]     = K_s_baseline;  // [uM.m]
      //workspace->y[N_Na_s]    = 0.00414271;
      workspace->y[N_Na_s]    = Na_s_baseline; // [uM.m]
      //workspace->y[N_HCO3_s]  = 0.000438336;
      workspace->y[N_HCO3_s]  = HCO3_s_baseline;
   }

   workspace->y[K_p]       = 3246.44;

   {//SMC
      workspace->y[ca_i]      = 0.137077;  // microMolar
      workspace->y[ca_sr_i]   = 1.20037;  // unit? (maybe mM)
      workspace->y[v_i]       = -58.5812; // [mV]
      workspace->y[w_i]       = 0.38778;
      workspace->y[ip3_i]     = 0.45; // uM
      workspace->y[K_i]       = 99994.8; // uM
   }

   {//EC (endothelial)
      //TODO: revise Ca_j why so big in EC?
      workspace->y[ca_j]      = 0.537991;  // uM 
      workspace->y[ca_er_j]   = 0.872007;  // uM
      workspace->y[v_j]       = -64.8638;  // mV
      workspace->y[ip3_j]     = 1.35;  // uM
   }

   //Mech (SMC)
   workspace->y[Mp]        = 0.0165439;
   workspace->y[AMp]       = 0.00434288;
   workspace->y[AM]        = 0.0623458;

   workspace->y[K_e]	= 2811.29; // (microMolar)

   // NO pathway*************//****
   workspace->y[NOn]       = 0.273264;
   workspace->y[NOk]       = 0.21676;
   workspace->y[NOi]       = 0.160269;
   workspace->y[NOj]       = 0.159001;
   workspace->y[cGMP]      = 11.6217;
   workspace->y[eNOS]      = 2.38751;
   workspace->y[nNOS]      = 0.317995;
   workspace->y[ca_n]      = 0.1;
   workspace->y[E_b]       = 0.184071;
   workspace->y[E_6c]      = 0.586609;
}

// right hand side evaluation function.
//      t       time,
//      u       state variables, the first of which is the vessel radius
//      du      output vector, in the same order (already allocated)
void NVUNode::nvu_rhs(double t, double *u, double *du, double *fluxes)
{
    double p = (*getSharedMembers().pressures)[workspace->pressuresIndex];
    assert(!isnan(p));
#if defined(SIMULATE_ISOLATE_VESSEL_CLAMP_PRESSURE)
    //clamp pressure
    p = 2.0; //unitless
#endif

    // Fluxes:

    // pressure
    fluxes[flu_pt] = P0 / 2 * (p + workspace->pcap); // x P0 to make dimensional, transmural pressure in Pa
    //std::cout<< "TUAN" << fluxes[flu_pt] <<":"<<p<<"__"; //around 4290.xx (and change), and p around 0.57xx (and change)

#if defined(SIMULATE_ISOLATE_VESSEL_CLAMP_PRESSURE)
    //fluxes[flu_pt] = 16000; // NVU (without parbrain) value.
#endif
    fluxes[flu_P_str] = fluxes[flu_pt] * PA2MMHG; // transmural pressure in mmHg
    fluxes[flu_delta_p] = P0 * (p - workspace->pcap); // dimensional pressure drop over leaf vessel
#if defined(SIMULATE_ISOLATE_VESSEL_CLAMP_PRESSURE)
    //fluxes[flu_delta_p] = 18.6; // NVU (without parbrain) value.
    //fluxes[flu_delta_p] = 18.6; // NVU (without parbrain) value.
#endif

    {// SC fluxes
    fluxes[flu_R_s] = R_tot - u[R_k];                            // u[R_k] is AC volume-area ratio, fluxes[flu_R_s] is SC

    fluxes[flu_N_Cl_s]         	= u[N_Na_s] + u[N_K_s] - u[N_HCO3_s];  //
    fluxes[flu_Na_s]           	= u[N_Na_s] / fluxes[flu_R_s];      //

    fluxes[flu_K_s]            	= u[N_K_s] / fluxes[flu_R_s];           //
    fluxes[flu_HCO3_s]         	= u[N_HCO3_s] / fluxes[flu_R_s];        //
    fluxes[flu_Cl_s]           	= fluxes[flu_N_Cl_s] / fluxes[flu_R_s];  //
    }

    {// AC fluxes
    fluxes[flu_E_TRPV_k] = R_gas * Temp / (z_Ca * Farad) * log(u[ca_p] / u[ca_k]); // TRPV4 channel Nernst Potential

    fluxes[flu_Na_k]           	= u[N_Na_k] / u[R_k];                     //
    fluxes[flu_K_k]            	= u[N_K_k] / u[R_k];                      //
    fluxes[flu_HCO3_k]         	= u[N_HCO3_k] / u[R_k];                   //
    fluxes[flu_Cl_k]           	= u[N_Cl_k] / u[R_k];                     //

    fluxes[flu_E_Na_k] = (R_gas * Temp) / (z_Na * Farad) * log(fluxes[flu_Na_s] / fluxes[flu_Na_k]);    // V

    fluxes[flu_E_K_k] = (R_gas * Temp) / (z_K  * Farad) * log(fluxes[flu_K_s] / fluxes[flu_K_k] );     // V
    fluxes[flu_E_Cl_k] = (R_gas * Temp) / (z_Cl * Farad) * log(fluxes[flu_Cl_s] / fluxes[flu_Cl_k]);    // V
    fluxes[flu_E_NBC_k] = (R_gas * Temp) / (z_NBC* Farad) * log((fluxes[flu_Na_s] * pow(fluxes[flu_HCO3_s],2))/(fluxes[flu_Na_k] * pow(fluxes[flu_HCO3_k],2)));     // V
    fluxes[flu_E_BK_k] = reverseBK + switchBK * ((R_gas * Temp) / (z_K  * Farad) * log(u[K_p] / fluxes[flu_K_k]));   // V
    fluxes[flu_J_NaK_k] = J_NaK_max * ( pow(fluxes[flu_Na_k],1.5) / ( pow(fluxes[flu_Na_k],1.5) + pow(K_Na_k,1.5) ) ) * ( fluxes[flu_K_s] / (fluxes[flu_K_s] + K_K_s) );    // uMm s-1
    fluxes[flu_v_k] = ( g_Na_k * fluxes[flu_E_Na_k] + g_K_k * fluxes[flu_E_K_k] + g_TRPV_k * u[m_k] * fluxes[flu_E_TRPV_k] + g_Cl_k * fluxes[flu_E_Cl_k] + g_NBC_k * fluxes[flu_E_NBC_k] + g_BK_k * u[w_k] * fluxes[flu_E_BK_k] - fluxes[flu_J_NaK_k] * Farad / unitcon ) / ( g_Na_k + g_K_k + g_Cl_k + g_NBC_k + g_TRPV_k * u[m_k] + g_BK_k * u[w_k] );
    fluxes[flu_J_KCC1_k] = (R_gas * Temp * g_KCC1_k) / (pow(Farad,2)) * log(((fluxes[flu_K_s]) * (fluxes[flu_Cl_s]))/((fluxes[flu_K_k])*(fluxes[flu_Cl_k]))) * unitcon;   //uMm s-1
    fluxes[flu_J_NBC_k] = g_NBC_k / Farad * (fluxes[flu_v_k] - fluxes[flu_E_NBC_k]) * unitcon;       //uMm s-1
    fluxes[flu_J_NKCC1_k] = (g_NKCC1_k * R_gas * Temp) / (pow(Farad,2))  * log(((fluxes[flu_K_s]) * (fluxes[flu_Na_s]) * pow(fluxes[flu_Cl_s],2)) /((fluxes[flu_K_k]) * (fluxes[flu_Na_k]) * pow(fluxes[flu_Cl_k],2)))*unitcon;        //uMm s-1
    fluxes[flu_J_Na_k]  = g_Na_k / Farad * (fluxes[flu_v_k] - fluxes[flu_E_Na_k]) * unitcon;              //uMm s-1
    fluxes[flu_J_K_k]   = g_K_k  / Farad * ((fluxes[flu_v_k]) - (fluxes[flu_E_K_k] )) * unitcon;          //uMm s-1
    fluxes[flu_J_BK_k]  = g_BK_k / Farad * u[w_k] * (fluxes[flu_v_k] - fluxes[flu_E_BK_k]) * unitcon;  //uMm s-1
    }

    {// SMC fluxes
    fluxes[flu_M] = 1 - u[Mp] - u[AM] - u[AMp];
    fluxes[flu_h_r] = 0.1 * state_r; //(non-dimensional!)
    fluxes[flu_v_cpl_i] = - g_hat * ( u[v_i] - u[v_j] );
    fluxes[flu_c_cpl_i]         = - p_hat * ( u[ca_i] - u[ca_j] );
    fluxes[flu_I_cpl_i]         = - p_hatIP3 * ( u[ip3_i] - u[ip3_j] );
    fluxes[flu_rho_i] = 1;
    fluxes[flu_ip3_i] = Fmax_i *  pow(u[ip3_i],2) / ( pow(Kr_i,2) + pow(u[ip3_i],2) );
    fluxes[flu_SRuptake_i]      = B_i * pow(u[ca_i],2) / ( pow(u[ca_i],2) + pow(cb_i,2) );
    fluxes[flu_CICR_i] = CICR_rate * pow(u[ca_sr_i],2) / ( pow(sc_i,2) + pow(u[ca_sr_i],2) ) *  ( pow(u[ca_i],4) ) / ( pow(cc_i,4) + pow(u[ca_i],4) );
    fluxes[flu_extrusion_i]     = D_i * u[ca_i] * (1 + ( u[v_i] - vd_i ) / Rd_i );
    fluxes[flu_leak_i]       = L_i * u[ca_sr_i];
    fluxes[flu_VOCC_i]      = G_Ca * ( u[v_i] - v_Ca1 ) / ( 1 + exp( - ( u[v_i] - v_Ca2 ) / ( R_Ca ) ) );
    fluxes[flu_NaCa_i]      = G_NaCa * u[ca_i] * ( u[v_i] - v_NaCa ) / ( u[ca_i] + c_NaCa ) ;
    fluxes[flu_NaK_i]      = F_NaK;
    fluxes[flu_Cl_i]      = G_Cl * (u[v_i] - v_Cl);
    fluxes[flu_K_i] = G_K * u[w_i] * ( u[v_i] - vK_i );

    fluxes[flu_degrad_i]	    = const_k_i * u[ip3_i];
    fluxes[flu_J_stretch_i] = G_stretch/(1 + exp( -alpha1 * (fluxes[flu_P_str] * state_r / fluxes[flu_h_r] - sig0))) * (u[v_i] - Esac);
    fluxes[flu_v_KIR_i] = z_1 * u[K_p] / unitcon + z_2; // mV u[K_p],
    fluxes[flu_G_KIR_i] = exp( z_5 * u[v_i] + z_3 * u[K_p] / unitcon + z_4 ); // pS pF-1 =s-1 u[v_i], u[K_p]
    fluxes[flu_J_KIR_i] = F_il/delta_mv * (fluxes[flu_G_KIR_i]) * (u[v_i] - (fluxes[flu_v_KIR_i])); // mV s-1 // u[v_i], u[K_p]
    }

    {// EC fluxes
    fluxes[flu_v_cpl_j]			= - g_hat * ( u[v_j] - u[v_i] );
    fluxes[flu_c_cpl_j]			= - p_hat * ( u[ca_j] - u[ca_i] );
    fluxes[flu_I_cpl_j]			= - p_hatIP3 * ( u[ip3_j] - u[ip3_i] );
    fluxes[flu_rho_j] 			= 1;
    fluxes[flu_O_j] 			= JO_j;
    fluxes[flu_ip3_j]			= Fmax_j * ( pow(u[ip3_j], 2) ) / ( pow(Kr_j, 2) + pow(u[ip3_j], 2) );
    fluxes[flu_ERuptake_j]      = B_j * ( pow(u[ca_j], 2) ) / ( pow(u[ca_j], 2) + pow(cb_j, 2) );
    fluxes[flu_CICR_j] = C_j *  ( pow(u[ca_er_j], 2) ) / ( pow(sc_j, 2) + pow(u[ca_er_j], 2) ) *  ( pow(u[ca_j], 4) ) / ( pow(cc_j,4) + pow(u[ca_j],4) );
    fluxes[flu_extrusion_j]     = D_j * u[ca_j];
    fluxes[flu_leak_j]          = L_j * u[ca_er_j];
    fluxes[flu_cation_j] = G_cat * ( E_Ca - u[v_j]) * 0.5 * ( 1 + tanh( ( log10( u[ca_j] ) - m3cat ) / m4cat ) );
    fluxes[flu_BKCa_j] = 0.2 * ( 1 + tanh( ( ( log10(u[ca_j]) - const_c) * ( u[v_j] - const_b ) - const_a1 ) / ( m3b* pow(( u[v_j] + a2 * ( log10( u[ca_j] ) - const_c ) - const_b),2) + m4b ) ) );
    fluxes[flu_SKCa_j] = 0.3 * ( 1 + tanh( ( log10(u[ca_j]) - m3s ) / m4s ));
    fluxes[flu_K_j] = G_tot * ( u[v_j] - vK_j ) * ( fluxes[flu_BKCa_j] + fluxes[flu_SKCa_j] );
    fluxes[flu_R_j] = G_R * ( u[v_j] - v_rest);
    fluxes[flu_degrad_j] = const_k_j * u[ip3_j];
    fluxes[flu_J_stretch_j] = G_stretch / (1 + exp(-alpha1*(fluxes[flu_P_str] * state_r / fluxes[flu_h_r] - sig0))) * (u[v_j] - Esac);
    }

    // Mech fluxes
    fluxes[flu_K1_c]    = gam_cross * pow(u[ca_i],3);
    fluxes[flu_K6_c]    = fluxes[flu_K1_c];
    fluxes[flu_F_r]	= u[AMp] + u[AM];

    fluxes[flu_E] 	= EPASSIVE + fluxes[flu_F_r] * (EACTIVE - EPASSIVE);
    fluxes[flu_R_0] 	= R_0_passive_k + fluxes[flu_F_r] * (RSCALE - 1) * R_0_passive_k;
    

    // NO pathway fluxes
#if INPUT_TO_NVU == OPTION_LIFE
    fluxes[flu_P_NR2AO]         = nvu_Glu(t) / (betA + nvu_Glu(t));
    fluxes[flu_P_NR2BO]         = nvu_Glu(t) / (betB + nvu_Glu(t));
#elif INPUT_TO_NVU == OPTION_NEURON
    fluxes[flu_P_NR2AO]         = Glut_out / (betA + Glut_out);
    fluxes[flu_P_NR2BO]         = Glut_out / (betB + Glut_out);
#endif

    fluxes[flu_I_Ca]            = (-4 * v_n * G_M * P_Ca_P_M * (Ca_ex/const_M)) / (1+exp(-80*(v_n+0.02))) * (exp(2 * v_n * Farad / (R_gas * Temp))) / (1 - exp(2 * v_n * Farad / (R_gas * Temp))) * (0.63 * fluxes[flu_P_NR2AO] + 11 * fluxes[flu_P_NR2BO]);
    fluxes[flu_CaM]             = u[ca_n] / const_m_c;            // concentration of calmodulin / calcium complexes ; (100)
    fluxes[flu_tau_w] = (R_0_passive_k * state_r) * fluxes[flu_delta_p] / (2*L0); // WSS using pressure from the H tree. L_0 = 200 um

    fluxes[flu_W_tau_w]         = W_0 * pow((fluxes[flu_tau_w] + sqrt(16 * pow(delt_wss,2) + pow(fluxes[flu_tau_w],2)) - 4 * delt_wss),2) / (fluxes[flu_tau_w] + sqrt(16 * pow(delt_wss,2) + pow(fluxes[flu_tau_w],2))) ;  // - tick
    fluxes[flu_F_tau_w]         = 1 / (1 + const_alp * exp(-fluxes[flu_W_tau_w])) - 1 / (1 + const_alp); // -(1/(1+alp)) was added to get no NO at 0 wss (!) - tick
    fluxes[flu_k4]              = C_4 * pow(u[cGMP], const_m);
    fluxes[flu_R_cGMP2]         = (pow(u[cGMP], 2)) / (pow(u[cGMP], 2) + pow(K_m_mlcp, 2)); // - tick
    fluxes[flu_K2_c] = 58.1395 * k_mlcp_b + 58.1395 * k_mlcp_c * fluxes[flu_R_cGMP2];  // Factor is chosen to relate two-state model of Yang2005 to Hai&Murphy model
    fluxes[flu_K5_c]            = fluxes[flu_K2_c];
    fluxes[flu_c_w]             = 1/2 * ( 1 + tanh( (u[cGMP] - 10.75) / 0.668 ) );

    fluxes[flu_Kactivation_i]   = pow((u[ca_i] + fluxes[flu_c_w]),2) / (pow((u[ca_i] + fluxes[flu_c_w]),2) + bet_i * exp(-(u[v_i] - v_Ca3) / R_K));
    fluxes[flu_E_5c]				= 1 - u[E_b] - u[E_6c];
    fluxes[flu_V_max_pde]			= k_pde * u[cGMP];

    // AC Ca2+ fluxes
#if INPUT_TO_NVU == OPTION_LIFE
    fluxes[flu_rho]				= 0.1 + 0.6/1846 * nvu_Glu(t);
#elif INPUT_TO_NVU == OPTION_NEURON
    fluxes[flu_rho]				= 0.1 + 0.6/1846 * Glut_out;
#endif

    fluxes[flu_ip3_k] 		= J_max * pow(( u[ip3_k] / ( u[ip3_k] + K_I ) * u[ca_k] / ( u[ca_k] + const_K_act ) * u[h_k] ) , 3) * (1.0 - u[ca_k] / u[s_k]);
    fluxes[flu_er_leak] 	= P_L * ( 1.0 - u[ca_k] / u[s_k] );
    fluxes[flu_pump_k]  	= const_V_max * pow(u[ca_k], 2) / ( pow(u[ca_k], 2) + pow(k_pump, 2) );
    fluxes[flu_I_TRPV_k]	= G_TRPV_k * u[m_k] * (fluxes[flu_v_k] - fluxes[flu_E_TRPV_k]) * unitcon;
    fluxes[flu_TRPV_k]		= -0.5 * fluxes[flu_I_TRPV_k] / ( C_astr_k * gamma_k );
    fluxes[flu_B_cyt] 			= 1.0 / (1.0 + BK_end + K_ex * B_ex / pow((K_ex + u[ca_k]), 2) );
    fluxes[flu_G]				= ( fluxes[flu_rho] + const_delta ) / ( K_G + fluxes[flu_rho] + const_delta );
    fluxes[flu_v_3]				= -v_5 / 2.0 * tanh((u[ca_k] - Ca_3) / Ca_4) + v_7;
    fluxes[flu_w_inf]    	= 0.5 * ( 1 + tanh( ( fluxes[flu_v_k] + eet_shift * u[eet_k] - fluxes[flu_v_3] ) / v_4 ) );
    fluxes[flu_phi_w]    	= psi_w * cosh( (fluxes[flu_v_k] - fluxes[flu_v_3]) / (2*v_4) );
    fluxes[flu_H_Ca_k]			= u[ca_k] / gam_cai_k + u[ca_p] / gam_cae_k;
    fluxes[flu_eta] 			= (state_r* R_0_passive_k - R_0_passive_k) / R_0_passive_k;
    fluxes[flu_minf_k] 			= ( 1 / ( 1 + exp( - (fluxes[flu_eta] - epshalf_k) / kappa_k ) ) ) * ( ( 1 / (1 + fluxes[flu_H_Ca_k]) ) * (fluxes[flu_H_Ca_k] + tanh(( fluxes[flu_v_k] - v1_TRPV_k) / v2_TRPV_k )));
    fluxes[flu_t_Ca_k] 			= t_TRPV_k / u[ca_p];
    fluxes[flu_VOCC_k]		= fluxes[flu_VOCC_i];
#if INPUT_TO_NVU == OPTION_LIFE
    //from presynaptic(firing) neuron only
    fluxes[flu_K_input]		= K_input(t);
    fluxes[flu_nvu_Glu]     = nvu_Glu(t);
#elif INPUT_TO_NVU == OPTION_NEURON
    fluxes[flu_K_input]		= J_K_out;
    fluxes[flu_nvu_Glu]     = J_Glut_out;
#endif

// Differential Equations:

    du[i_radius		] = 1 / ETA * (state_r * fluxes[flu_pt] / fluxes[flu_h_r] - fluxes[flu_E] * (state_r * R_0_passive_k - fluxes[flu_R_0])/fluxes[flu_R_0]); // Radius - nondimensional (state_r * R_0_passive: dimensional)

    //AC:
    du[ R_k     ] = L_p * (fluxes[flu_Na_k] + fluxes[flu_K_k] + fluxes[flu_Cl_k] + fluxes[flu_HCO3_k] - fluxes[flu_Na_s] - fluxes[flu_K_s] - fluxes[flu_Cl_s] - fluxes[flu_HCO3_s] + X_k / u[R_k]);  // m s-1

    du[ N_Na_k  ] = -fluxes[flu_J_Na_k] - 3 * fluxes[flu_J_NaK_k] + fluxes[flu_J_NKCC1_k] + fluxes[flu_J_NBC_k];    // uMm s-1
    du[ N_K_k   ] = -fluxes[flu_J_K_k] + 2 * fluxes[flu_J_NaK_k] + fluxes[flu_J_NKCC1_k] + fluxes[flu_J_KCC1_k] -fluxes[flu_J_BK_k]; // uMm s-1
    du[ N_HCO3_k] = 2 * fluxes[flu_J_NBC_k];                                                // uMm s-1
    du[ N_Cl_k  ] = du[N_Na_k] + du[ N_K_k] - du[ N_HCO3_k]; // uMm s-1, modified equation compared to the one of Ostby  //
    du[ w_k     ] = fluxes[flu_phi_w] * (fluxes[flu_w_inf] - u[w_k]);                            // s-1

    //SC:
#if INPUT_TO_NVU == OPTION_LIFE
    du[N_Na_s] = - k_C * K_input(t) - du[N_Na_k]; // uMm s-1
    du[N_K_s] = k_C * K_input(t) + fluxes[flu_J_K_k] - 2 * fluxes[flu_J_NaK_k] - fluxes[flu_J_NKCC1_k] - fluxes[flu_J_KCC1_k] + fluxes[flu_R_s] * ( (u[K_e] - fluxes[flu_K_s]) / tau2);                 // uMm s-1
#elif INPUT_TO_NVU == OPTION_NEURON
    /* assuming
     * Na uptake into (presynaptic/firing) neuron
     * K+ release from (presynaptic/firing) neuron
     * are of the same rate (k_C)
     * and the flux J_K_out, J_Na_out
     */

    /*
     * During neuronal activity, outflux of K+ is counter balance by the same amount in terms of
     *    total influx of Na+ and Cl-
     * Here, Ostby 1999 assume the majority of influx is via Na+, so
     * J_Na is about (-1 * J_K)
     */
    J_Na_out = J_K_out;
    du[N_Na_s] = - k_C * J_Na_out - du[N_Na_k];                           // uMm s-1
    du[N_K_s] =
       /*neuron*/ k_C * J_K_out
       /*astrocyte*/ + fluxes[flu_J_K_k] - 2 * fluxes[flu_J_NaK_k] - fluxes[flu_J_NKCC1_k] - fluxes[flu_J_KCC1_k]
       /*SC*/ + fluxes[flu_R_s] * ( (u[K_e] - fluxes[flu_K_s]) / tau2);                 // uMm s-1
#endif


    du[ N_HCO3_s] = - du[ N_HCO3_k];                                                // uMm s-1

    //PVS:
    du[ K_p     ] = fluxes[flu_J_BK_k] / (VR_pa * u[R_k]) + fluxes[flu_J_KIR_i] / VR_ps - R_decay * (u[K_p] - K_p_min);         // uM s-1

    {//SMC:
    du[ ca_i    ] = fluxes[flu_c_cpl_i] + fluxes[flu_rho_i] * ( fluxes[flu_ip3_i] - fluxes[flu_SRuptake_i] + fluxes[flu_CICR_i] - fluxes[flu_extrusion_i] + fluxes[flu_leak_i] - fluxes[flu_VOCC_i] + fluxes[flu_NaCa_i] - 0.1* fluxes[flu_J_stretch_i]);
    du[ ca_sr_i ] = fluxes[flu_SRuptake_i] - fluxes[flu_CICR_i] - fluxes[flu_leak_i] ;
    du[ v_i     ] = fluxes[flu_v_cpl_i] + delta_mv * ( - fluxes[flu_NaK_i] - fluxes[flu_Cl_i] - 2 * fluxes[flu_VOCC_i] - fluxes[flu_NaCa_i] - fluxes[flu_K_i] - fluxes[flu_J_stretch_i] - fluxes[flu_J_KIR_i] );
    du[ w_i     ] = lam * (fluxes[flu_Kactivation_i] - u[w_i] ) ;
    du[ ip3_i   ] = fluxes[flu_I_cpl_i] - fluxes[flu_degrad_i] ;          // **
    du[ K_i     ] = - fluxes[flu_J_KIR_i] - fluxes[flu_K_i] + fluxes[flu_NaK_i];                                            // uM s-1
    // Mech: (SMC)
    du[ Mp   	] = K4_c * u[AMp] + fluxes[flu_K1_c] * fluxes[flu_M] - (fluxes[flu_K2_c] + K3_c) * u[Mp];
    du[ AMp  	] = K3_c * u[Mp] + fluxes[flu_K6_c] * u[AM] - (K4_c + fluxes[flu_K5_c]) * u[AMp];
    du[ AM   	] = fluxes[flu_K5_c] * u[AMp] - ( K7_c + fluxes[flu_K6_c] ) * u[AM];
    }

    {//EC:
    du[ca_j     ] = fluxes[flu_c_cpl_j] + fluxes[flu_rho_j] * ( fluxes[flu_ip3_j] - fluxes[flu_ERuptake_j] + fluxes[flu_CICR_j] - fluxes[flu_extrusion_j] + fluxes[flu_leak_j] + fluxes[flu_cation_j] + fluxes[flu_O_j] - fluxes[flu_J_stretch_j] ) ;
    du[ca_er_j  ] = fluxes[flu_ERuptake_j] - fluxes[flu_CICR_j] - fluxes[flu_leak_j] ;
    du[v_j      ] = fluxes[flu_v_cpl_j] - 1/C_m * ( fluxes[flu_K_j] + fluxes[flu_R_j] ) ;
    du[ip3_j    ] = fluxes[flu_I_cpl_j] + J_PLC - fluxes[flu_degrad_j] ;  // **

    }

    {//ECS:		smc efflux				SC flux
    du[K_e] = - fluxes[flu_NaK_i] + fluxes[flu_K_i] - ( (u[K_e] - fluxes[flu_K_s]) / tau2) + ECS_input(t);
    }
    /***********NO pathway***********/

    {// NE:
    // TODO: remove that and bring to NTS's neuron model
    //  or to MegaSynapticSpace
    du[ca_n]       = (fluxes[flu_I_Ca] / (2*Farad * V_spine) - (k_ex * (u[ca_n] - Ca_rest))) / (1 + lambda);     //\muM
    du[nNOS]       = V_maxNOS * fluxes[flu_CaM] / (K_actNOS + fluxes[flu_CaM]) - mu2 * u[nNOS] ;                  //\muM
    du[NOn]       = u[nNOS] * V_max_NO_n * const_On / (K_mO2_n + const_On) * LArg / (K_mArg_n + LArg) - ((u[NOn] - u[NOk]) / tau_nk) - (k_O2* pow(u[NOn],2) * const_On);
    }

    // AC:
    du[NOk]       = (u[NOn] - u[NOk]) / tau_nk + (u[NOi] - u[NOk]) / tau_ki - k_O2 * pow(u[NOk],2) * const_Ok;

    // SMC:
    du[NOi]       = (u[NOk] - u[NOi]) / tau_ki + (u[NOj] - u[NOi]) / tau_ij - k_dno * u[NOi] ;
    du[E_b]        = -k1 * u[E_b] * u[NOi] + k_1 * u[E_6c] + fluxes[flu_k4] * fluxes[flu_E_5c];
    du[E_6c]       = k1 * u[E_b] * u[NOi] - k_1 * u[E_6c] - k2 * u[E_6c] - k3 * u[E_6c] * u[NOi] ;
    du[cGMP]       = V_max_sGC * fluxes[flu_E_5c] - fluxes[flu_V_max_pde] * u[cGMP] / (K_m_pde + u[cGMP]);

    {// EC:
    du[eNOS]       = gam_eNOS * (K_dis * u[ca_j] / (K_eNOS + u[ca_j]))  // Ca-dependent activation - tick
					+ (1 - gam_eNOS) * (g_max * fluxes[flu_F_tau_w]) // wss-dependent activation - tick
					- mu2 * u[eNOS];      // deactivation - tick
    du[NOj]       = V_NOj_max * u[eNOS] * const_Oj / (K_mO2_j + const_Oj) * LArg_j / (K_mArg_j + LArg_j) // production - tick
					- k_O2 * pow(u[NOj],2) * const_Oj // consumption
					+ (u[NOi] - u[NOj]) / tau_ij - u[NOj] * 4 * D_NO / (pow(25,2)); // Either R0*state_r or 25
    }

    /**********Astrocytic Calcium*******/

    {// AC:
    du[ca_k]	= fluxes[flu_B_cyt] * (fluxes[flu_ip3_k] - fluxes[flu_pump_k] + fluxes[flu_er_leak] + fluxes[flu_TRPV_k]/r_buff);
    du[s_k]		= -(fluxes[flu_B_cyt] * (fluxes[flu_ip3_k] - fluxes[flu_pump_k] + fluxes[flu_er_leak])) / (VR_ER_cyt);
    du[h_k]		= k_on * (K_inh - (u[ca_k] + K_inh) * u[h_k]);
    du[ip3_k]	= r_h * fluxes[flu_G] - k_deg * u[ip3_k];
    du[eet_k]	= V_eet * fmax((u[ca_k] - Ca_k_min), 0) - k_eet * u[eet_k];
    du[m_k]		= trpv_switch * (fluxes[flu_minf_k] - u[m_k]) / (fluxes[flu_t_Ca_k] * u[ca_p]);
    }

    {//PVS:
    du[ca_p]	= (-fluxes[flu_TRPV_k] / VR_pa) + (fluxes[flu_VOCC_k] / VR_ps) - Ca_decay_k * (u[ca_p] - Capmin_k);
    }
}

double NVUNode::factorial(int c)
{
    double result = 1;

    for (int n = 1; n <= c; n++)
    {
        result = result * n;
    }

    return result;
}

#if INPUT_TO_NVU == OPTION_LIFE
// time-varying glutamate input signal
double NVUNode::nvu_Glu(double t)
{
    double Glu_min = 0;
    double Glu_max = 1846; // uM - one vesicle (Santucci)

    if (value == 0 || t < INITIAL_WAIT)
    {
        return 0.0;
    }

    double Glu_time = 0.5 * tanh((t - workspace->injectStart) / 1) - 0.5 * tanh((t - (workspace->injectStart + LIFE_TIME)) / 1);

    double Glu_out = Glu_min + (Glu_max - Glu_min) * Glu_time;
    return Glu_out;
}
#elif INPUT_TO_NVU == OPTION_NEURON
// time-varying glutamate input signal
// TODO: use LFP (from MegaSynapticSpace) and produce Glu
void NVUNode::nvu_Glu()
{
   double prev_Glut_out = Glut_out;
    if (*numSpikes >= THRESHOLD_NUMSPIKES)
    {
       //double Glu_min = 0;
       double Glut_max = 1846; // uM - one vesicle (Santucci)

       Glut_out += Glut_max;
    }
    //float Glut_lifetime = 10e-3; // sec
    float Glut_lifetime = 2e-3; // sec
    float J_decay = 1.0 / (Glut_lifetime) *
                    (Glut_out - GLUT_MIN);  // [uM/sec]
    Glut_out = std::max(GLUT_MIN, Glut_out - TStep * J_decay);
    J_Glut_out = (Glut_out - prev_Glut_out)/TStep;
}
#endif

#if INPUT_TO_NVU == OPTION_LIFE
// time-varying K+ input signal (simulating neuronal activity)
/*
 * Ostby model
 */
double NVUNode::K_input(double t)
{
    double K_input_min  = 0;
    double K_input_max  = 2.67; // unit??

    double lengthpulse  = LIFE_TIME / 2.0; // Half up then half down during 'pulse'
    double lengtht1     = INJECT_TIME;

    double t0           = workspace->injectStart;
    double t1           = t0 + lengtht1;
    double t2           = t0 + lengthpulse;
    double t3           = t1 + lengthpulse;

    int alpha           = 2;  // (unitless) alpha-distribution constant
    int beta            = 5;  // (unitless) beta-distribution constant
    double deltat       = INJECT_TIME;
    double gab          = factorial(alpha + beta - 1);
    double ga           = factorial(alpha - 1);
    double gb           = factorial(beta - 1);
    //TODO: switch to using factorial in NumberUtils
    //     and only store result of gab/(ga*gb)


    double K_time;

    //NOTE: currently use 'value' from LifeNode as the driver for neuron's Glut release
    if (value == 0 || t < INITIAL_WAIT)
    {
        return 0.0;
    }

    double Finput = 1.0; //(amplitude) scaling factor
    if (t >= t0 && t <= t1)
    {
        K_time = Finput * gab / (ga * gb) * 
	    pow((1 - (t - t0) / deltat), (beta - 1)) * 
	    pow(((t - t0) / deltat), (alpha-1)); //unitless
    }
    else if (t >= t2 && t <= t3)
    {
        K_time = - Finput;
        //K_time = - 1;
    }
    else
    {
        K_time = 0;
    }

    double K_out = K_input_min + (K_input_max - K_input_min) * K_time; // 0 if t3 < t or x,y <= 0
    return K_out;
}
#elif INPUT_TO_NVU == OPTION_NEURON
void NVUNode::nvu_K()
{
   //double prev_K_out = K_out;
    if (*numSpikes >= THRESHOLD_NUMSPIKES)
    {
	double K_input_max  = 2.67; // uM???
	float scale_factor = 10000.0;
	J_K_out = (K_input_max)/TStep/scale_factor;
    }
    float K_lifetime = 2e-3; // sec
    float J_decay = 1.0 / (K_lifetime) *
                    (workspace->y[N_K_s] - K_s_baseline);  // [uM/sec or mM/sec ??? try to find out]
    workspace->y[N_K_s] = std::max(K_s_min, workspace->y[N_K_s] - TStep * J_decay);

//    if (J_K_out > 0.0)
//       std::cerr << "J_K_out " << J_K_out << ", and *k_C " << J_K_out * k_C
//	  << "  numSpi " << *numSpikes << std::endl;
}
#endif

// time-varying PLC input signal
double NVUNode::PLC_input(double t)
{
    double PLC_min = 0.18;
    double PLC_max = 0.4;
    // double t_up   = 4000;
    // double t_down = 6000;
    double ampl = 3;
    double ramp = 0.003;//0.002;
    double PLC_time = 0.5 * tanh((t - workspace->injectStart) / 0.05) - 0.5 * tanh((t - (workspace->injectStart + LIFE_TIME / 2.0)) / 0.05);

    double PLC_out = PLC_min + (PLC_max-PLC_min) * PLC_time;
    return PLC_out;
}

double NVUNode::ECS_input(double t)
{
    double ECS_max 		= 9e3;
    double t_up   		= 1100;
    double t_down 		= 2000;
    double lengthpulse 	= t_down - t_up;
    double lengtht1 	= 20;
    double t0 			= t_up;
    double t1 			= t0 + lengtht1;
    double t2 			= t0 + lengthpulse;
    double t3 			= t1 + lengthpulse;

    double ampl = 3;
    double ramp = 0.003;
    double x_centre = 0;
    double y_centre = 0;

    double ECS_time;
    if (t >= t0 && t <= t1)
    {
        ECS_time = 1;
    }
    else if (t >= t2 && t <= t3)
    {
    	ECS_time = - 1;
    }
    else
    {
    	ECS_time = 0;
    }

    double ECS_out = ECS_max * ECS_time;

    return ECS_out;
}


void NVUNode::malloc_workspace()
{
    workspace = 0;
    workspace = (Workspace *)malloc(sizeof(Workspace));

    workspace->dfdx=0;
    workspace->J=0;

    workspace->dfdx_pattern = 0;

    workspace->N = 0;
    workspace->S = 0;
    workspace->y = 0;
    workspace->f = 0;
    workspace->fluxes = 0;
    workspace->state_var_file = 0;
    workspace->fluxes_file = 0;
    workspace->beta = 0;
    workspace->w = 0;
    workspace->x = 0;
}


void NVUNode::solver_init()
{
    malloc_workspace();     // allocate memory for all arrays, structs, and classes inside Workspace.

    workspace->t0     = 0.0;             // initial time 0
    workspace->t = workspace->t0;
    workspace->ftol   = 1e-3;           // function evaluation tolerance for Newton convergence 1e-3
    workspace->ytol   = 1e-3;           // relative error tolerance for Newton convergence 1e-3
    workspace->nconv  = 5;              // Newton iteration threshold for Jacobian reevaluation 5
    workspace->maxits = 100;            // Maximum number of Newton iterations 100
    //workspace->dtwrite = 1.0;             // Time step (in seconds) for writing to file (and screen)
    workspace->dtwrite = 0.10;             // Time step (in seconds) for writing to file (and screen)

    workspace->mdeclared = 0;

    workspace->jacupdates = 0;
    workspace->fevals     = 0;
    workspace->isjac = 0;

    //nvu init
    workspace->num_fluxes = 103;
    workspace->counter = 0;  // to be used in LIFE only
    workspace->injectStart = 0.0;  // for impulse simulation purpose, i.e. track when an impulse is given


    // Sparsity patterns (approximated by matrices full of 1s). Will be improved as simulation progresses.
    int dfdx_pattern[NEQ*NEQ];
    for(int i = 0; i < NEQ * NEQ ; dfdx_pattern[i++] = 1);

    // Takes the pattern defined above and puts it into a sparse matrix (cs*)
    cs *T;
    T = (cs *)cs_spalloc(NEQ, NEQ, 1, 1, 1);

    for (int j = 0; j < NEQ; j++)
    {
        for (int i = 0; i < NEQ; i++)
        {
            if (dfdx_pattern[NEQ*j + i])
            {
                cs_entry(T, i, j, 1.0);
            }
        }
    }

    workspace->dfdx_pattern = cs_compress(T);
    cs_spfree(T);


    workspace->pcap  = PCAP / P0; // pressure is nondimensional

    //workspace->l  = 1; // normalised away


    // Init jacobians
    workspace->nblocks = 1;


    //init_dgdx();
    //init_dpdg();
    init_dfdx();
    //init_dfdp();


    // Put initial conditions in to y
    workspace->y = zerosv(NEQ); // state vars vector
    workspace->f = zerosv(NEQ); // derivatives vector
    workspace->fluxes = zerosv(workspace->num_fluxes);

    nvu_ics(); // Set inital conditions.

}


css * NVUNode::newton_sparsity(cs *J)
{
    // Perform symbolic analysis of the Jacobian for subsequent LU
    // factorisation
    css *S;
    int order = 0;           // 0 = natural, 1 = min deg order of A + A'
    S = cs_sqr(order, J, 0); // 0 means we're doing LU and not QR
    return S;
}

void NVUNode::newton_matrix()
{
    // Create a Newton matrix from the given step gamma and Jacobian in W
    cs *M, *eye;
    if (workspace->mdeclared)
    {
        cs_nfree(workspace->N);
    }
    else
    {
        workspace->mdeclared = 1;
    }

    eye = speye(workspace->J->m);
    M = cs_add(eye, workspace->J, 1, -TStep);
    cs_spfree(eye);

    workspace->N = cs_lu(M, workspace->S, 1);
    cs_spfree(M);
}

// b  = array of all ODEs, i.e. 'du' array
int NVUNode::lusoln(double *b)
{
    // Can only be called if newton_matrix has been called already
    double *x;
    int n = workspace->J->n;
    x = (double *)cs_malloc (n, sizeof (*x));
    int ok = workspace->S && workspace->N && x;

    if (ok)
    {
        cs_ipvec(workspace->N->pinv, b, x, n);
        cs_lsolve(workspace->N->L, x);
        cs_usolve(workspace->N->U, x);
        cs_ipvec(workspace->S->q, x, b, n);
    }

    cs_free (x);

    return ok;
}

/**
This function is responsible for calculating the derivatives at time t.
It then calculates diffusion, updates the derivatives, and checks all
state variables, fluxes, and derivatives for NaN values.
*/
void NVUNode::evaluate(double t, double *y, double *dy)
{
    workspace->fevals++;
    nvu_rhs(t, y, dy, workspace->fluxes);
#ifdef MODEL_DIFFUSE
    diffuse(); 
#endif

    // Ensure all state variables, fluxes, and derivatives are non-NaN.
    for (int i = 0; i < NEQ; i++)
    {
	if (isnan(y[i]))
	{
	   std::cerr << " time " << (getSimulation().getIteration() * TStep)
	      << ": NaN for y[" << i << "]";
	}
        assert(!isnan(y[i]));
	if (isnan(dy[i]))
	{
	   std::cerr << " time " << (getSimulation().getIteration() * TStep)
	      << ": NaN for dy[" << i << "]";
	}
        assert(!isnan(dy[i]));
    }

    for (int i = 0; i < workspace->num_fluxes; i++)
    {
        assert(!isnan(workspace->fluxes[i]));
    }
}

/*
 * t <---- workspace->t   (time)
 * u <---- workspace->y   (state-Vars)
 */
void NVUNode::jacupdate(double t, double *u)
{
    workspace->jacupdates++;
    double *f;
    double eps = 1e-6;

    f = (double *)malloc(NEQ * sizeof (*f ));
    evaluate(t, u, f);  // f ~ du

    eval_dfdx(t, u, f, eps);

    if (workspace->isjac) cs_spfree(workspace->J);

    workspace->J = matcopy(workspace->dfdx->A);
    workspace->isjac = 1;

    free(f);
}

void NVUNode::init_dfdx()
{
    // Load sparsity pattern for one block (dfdx_pattern) and add to make Jacobian for all blocks (dfdx)
    int nblocks = workspace->nblocks;
    cs *J;
    J = blkdiag(workspace->dfdx_pattern, nblocks, nblocks);

    workspace->dfdx = numjacinit(J);

    cs_spfree(J);
}

/*
 * Evaluate
 * y = state-vars
 * f = dy
 * eps = epsilon
 */
void NVUNode::eval_dfdx(double t, double *y, double *f, double eps)
{
    int i, j;
    double *y1, *h, *f1;

    y1 = (double *)malloc((NEQ+1) * sizeof (*y1));
    h  = (double *)malloc((NEQ+1) * sizeof (*h));
    f1 = (double *)malloc((NEQ+1) * sizeof (*f1));

    for (int igrp = 0; igrp < workspace->dfdx->ng; igrp++)
    {
        for (int k = 0; k < NEQ; k++)
        {
            y1[k] = y[k];
        }

        for (int k = workspace->dfdx->r[igrp]; k < workspace->dfdx->r[igrp + 1]; k++)
        {
            j = workspace->dfdx->g[k];
            h[j] = eps; // * fabs(y[j]);
            y1[j] += h[j];
        }

        nvu_rhs(t,y1,f1, workspace->fluxes);

        for (int k = workspace->dfdx->r[igrp]; k < workspace->dfdx->r[igrp+1]; k++)
        {
            j = workspace->dfdx->g[k];

            for (int ip = workspace->dfdx->A->p[j]; ip < workspace->dfdx->A->p[j+1]; ip++)
            {
                i = workspace->dfdx->A->i[ip];
                workspace->dfdx->A->x[ip] = (f1[i] - f[i]) / h[j];
            }
        }
    }

    free(y1);
    free(h);
    free(f1);
}


/*
These two functions are for plotting purposes - they write out state variables and fluxes for a single
NVU node as .csv files, to be converted later using the plotting script "plot_species.py".

*/
void NVUNode::write_state_var_data()
{
    fprintf(workspace->state_var_file, "%f",workspace->t);
    for (int i = 0; i < NEQ; i++)
    {
    	// workspacerite radius values in um - they are non-dimensional in the system.
    	if (i == i_radius) fprintf(workspace->state_var_file,", %f",20.0 * workspace->y[i]);
    	else fprintf(workspace->state_var_file,", %.10f",workspace->y[i]);
    }
    fprintf(workspace->state_var_file, "\n");
}

void NVUNode::write_fluxes_data()
{
    fprintf(workspace->fluxes_file, "%f",workspace->t);
    for (int i = 0; i < workspace->num_fluxes; i++)
    {
        fprintf(workspace->fluxes_file,", %.10f",workspace->fluxes[i]);
    }
    fprintf(workspace->fluxes_file, "\n");
}

/**
Iterate over all connected NVU neighbors and calculate diffusion between them.
Updates own ecs K+ derivative value, expected to take place before solving for a given iteration.
// IN NVU 1.2 only ECS K+ is diffused.
*/
void NVUNode::diffuse()
{
    double flu_diff_k;
    for (int i = 0; i < K_ecs_neighbors.size(); i++)
    {
        flu_diff_k = (*(K_ecs_neighbors[i]) - workspace->y[K_e]) / tau_diffusion;
        workspace->f[K_e] += flu_diff_k;
    }
}


/**
Calculates the correct index to access its corresponding pressure from the H-tree
given its own coordinates and the coordinates of all end-terminals from the H-tree.
*/
void NVUNode::calculate_pressure_index()
{
    std::vector<int> gridSize =
	this->getGridLayerDescriptor()->getGrid()->getSize();
    int numberNVUs = gridSize[0] * gridSize[1] * gridSize[2];

     // Put coords values into a double array for use with matrixOps function later.
    double *myCoords = (double *)malloc(DIMENSIONS * sizeof(double*));
    for (int i = 0; i < DIMENSIONS; i++)
    {
        myCoords[i] = coords[i];
    }

    double *treeCoords = (double *)malloc(DIMENSIONS * sizeof(double*));

    // Get first coordinate from array to initalise shortest distance.
    for (int i = 0; i < DIMENSIONS; i++)
    {
        treeCoords[i] = (*getSharedMembers().coordsArray)[i];
    }
    double shortestDistance;
    double distance;
    shortestDistance = distanceBetween2Points(myCoords, treeCoords, DIMENSIONS);

    workspace->pressuresIndex = 0;
    int currentNodeIndex = 0;
    int bestIndex = 0;

    // Find where it's found inside getSharedMembers().coordsArray with information of DIMENSIONS.
    // Begin at second set of coordinates.
    for (int i = DIMENSIONS; i < numberNVUs * DIMENSIONS; i += DIMENSIONS)
    {
        // Increment first because we've already looked at the first item.
        currentNodeIndex++;

        // fill theirCoords
        for (int j = 0; j < DIMENSIONS; j++)
        {
            treeCoords[j] = (*getSharedMembers().coordsArray)[i + j];
        }

        distance = distanceBetween2Points(myCoords, treeCoords, DIMENSIONS);
        if (distance < shortestDistance)
        {
            shortestDistance = distance;
            bestIndex = currentNodeIndex;
        }
    }

    // Relationship between coordinate and pressure arrays in H-tree. No dependency between nodes.
    workspace->pressuresIndex = bestIndex / 2;
#ifdef DEBUG
    //printf("Node: %d. P-index: %d\n", getNodeIndex(), workspace->pressuresIndex);
    int rank = getSimulation().getRank();
    int comm_size = getSimulation().getNumProcesses();
    for (int i = 0; i < comm_size; i++)
    {
       MPI_Barrier(MPI_COMM_WORLD);
       std::vector<int> coords;
       int x, y;
       getNodeCoords(coords);
       //getNodeCoords2Dim(x, y);
       if (i == rank and getSimulation().isSimulatePass())
       {
	  char str[1024];
	  snprintf(str, sizeof(str), "rank: %d, Node: %d. P-index: %d\n"
		, i, getNodeIndex(), workspace->pressuresIndex);
	  std::cerr << str << std::endl; 
       }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    free(myCoords);
    free(treeCoords);
}

NVUNode::~NVUNode()
{
}

/*
 * some times we want to export NVU-associated data, but these
 * data are not solved by NVU's ODE system
 * condition(int):   0 = add element to array
 *                   1 = assign data to array
 */
void NVUNode::add_or_update_extra_stateVariables(int condition)
{
    // Glut concentration
    // numSpikes
    if (condition == 0)
    {//add elements
       assert(stateVariables.size() == NEQ);
       stateVariables.push_back(Glut_out);
       /* these already in ODE system
       // K+ extracellular
       stateVariables.push_back(K_out);
	*/
#if INPUT_TO_NVU == OPTION_NEURON
       stateVariables.push_back(*numSpikes);
#else
       stateVariables.push_back(0.0);
#endif
    }
    else{
       //assign value
       //stateVariables[NEQ] = K_out;
       stateVariables[NEQ] = Glut_out;
#if INPUT_TO_NVU == OPTION_NEURON
       stateVariables[NEQ+1] = *numSpikes;
#endif
    }
}
