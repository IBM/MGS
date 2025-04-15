// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "MotoneuronUnit.h"
#include "CG_MotoneuronUnit.h"
#include "rndm.h"

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

#define F 96485.332 // Faraday constant, C / molar
#define R 8.31450 // gas constant, in J / molar
 
void MotoneuronUnit::initialize(RNG& rng)
{
  // ##### WHOLE NEURON #####
  // Geometry - scale the average to this neuron's individual versions
  assert(SHD.d_diameter.size()==3);  
  assert(SHD.d_segmentsProp.size()==3);
  ind_d_length.increaseSizeTo(3);
  ind_d_diameter.increaseSizeTo(3);
  ind_d_surfaceArea.increaseSizeTo(3);  
  for (int i=0; i < 3; ++i) // going proximal through to distal
    {
      ind_d_length[i] = (SHD.d_length * ind_size) * SHD.d_segmentsProp[i];
      ind_d_diameter[i] = SHD.d_diameter[i] * ind_size;
      ind_d_surfaceArea[i] = ( ( 2.0 * M_PI * (ind_d_diameter[i] / 2.0) *
                               ind_d_length[i] )
        + ( 2.0 * M_PI * pow(ind_d_diameter[i]/2.0,2) )
                               ) * 0.00000001;
    }

  ind_s_diameter = SHD.s_diameter * ind_size;
  ind_s_surfaceArea = (4.0 * M_PI * pow(ind_s_diameter/2.0,2)
                       ) * 0.00000001;
  
  ind_i_length = SHD.i_length * ind_size;
  ind_i_diameter = SHD.i_diameter * ind_size;

  /*
  assert(SHD.a_nodeN==SHD.a_FLUT_STIN_length.size());
  ind_a_node_length = SHD.a_node_length * ind_size;
  ind_a_node_diameter = SHD.a_node_diameter * ind_size;
  ind_a_MYSA_length = SHD.a_MYSA_length * ind_size;
  ind_a_MYSA_diameter = SHD.a_MYSA_diameter * ind_size;
  ind_a_FLUT_STIN_length.increaseSizeTo(SHD.a_nodeN);
  for (int i=0; i < SHD.a_nodeN; ++i)
    ind_a_FLUT_STIN_length[i] = SHD.a_FLUT_STIN_length[i] * ind_size; 
  ind_a_FLUT_STIN_diameter = SHD.a_FLUT_STIN_diameter * ind_size;
  */
  
  // ##### DENDRITE #####
  d_V_m.increaseSizeTo(2);
  d_I_leak.increaseSizeTo(2);
  for (int i=0; i < 2; ++i)
    d_V_m[i] = SHD.V_rest;
  d_V_m_last = SHD.V_rest;
  
  // ##### SOMA #####
  s_V_m = SHD.V_rest;
  // Maximum conductance of ion channels
  ind_s_g_Naf = SHD.s_g_Naf * ind_size;
  ind_s_g_Kdr = SHD.s_g_Kdr * ind_size;
  ind_s_g_CaN = SHD.s_g_CaN * ind_size;
  ind_s_g_CaL = SHD.s_g_CaL * ind_size;
  ind_s_g_KCa = SHD.s_g_KCa * ind_size;
  // Fast sodium current
  double a = (0.4*(-(s_V_m+66.0)))/(exp(-(s_V_m+66.0)/5.0)-1.0);
  double b = (0.4*(s_V_m+32.0))/(exp((s_V_m+32.0)/5.0)-1.0);
  s_m_Naf = a / (a + b);
  s_h_Naf = 1.0/(1.0+exp((s_V_m+65.0)/7.0));
  // Delayed rectifier potassium current
  s_n_Kdr = 1.0/(1.0+exp((s_V_m+38.0)/-15.0));
  // N-type calcium current
  s_m_CaN = 1.0/(1.0+exp((s_V_m+32.0)/-5.0));
  s_h_CaN = 1.0/(1.0+exp((s_V_m+50.0)/5.0));
  // L-type calcium current
  s_p_CaL = 1.0/(1.0+exp((s_V_m+55.8)/-3.7));

  // ##### IAS #####
  i_V_m = SHD.V_rest;
  // Maximum conductance of ion channels
  ind_i_g_Naf = SHD.i_g_Naf * ind_size;
  ind_i_g_Nap = SHD.i_g_Nap * ind_size;
  ind_i_g_Kdr = SHD.i_g_Kdr * ind_size;
  // Fast sodium current
  a = (0.4*(-(i_V_m+60.0)))/(exp(-(i_V_m+60.0)/5.0)-1.0);
  b = (0.4*(i_V_m+40.0))/(exp((i_V_m+40.0)/5.0)-1.0);
  i_m_Naf = a / (a + b);
  i_h_Naf = 1.0/(1.0+exp((s_V_m+65.0)/7.0));
  // Persistent sodium current
  a = (0.0353*(i_V_m+28.4))/(1.0-exp(-(i_V_m+28.4)/5.0));
  b = (0.000883*(-(i_V_m+32.7)))/(1-exp((i_V_m+32.7)/5.0));
  i_p_Nap = a / (a + b);;
  // Delayed rectifier potassium current
  i_n_Kdr = 1.0/(1.0+exp((i_V_m+38.0)/-15.0));

  /*
  // ##### AXON #####
  // Maximum conductance of ion channels
  ind_a_g_Naf = SHD.a_g_Naf * ind_size;
  ind_a_g_Nap = SHD.a_g_Nap * ind_size;
  ind_a_g_Ks = SHD.a_g_Ks * ind_size;
  ind_a_g_MYSA = SHD.a_g_MYSA * ind_size;
  ind_a_g_FLUT_STIN = SHD.a_g_FLUT_STIN * ind_size;
  a_V_m_node.increaseSizeTo(SHD.a_nodeN);  
  a_V_m_MYSA_1.increaseSizeTo(SHD.a_nodeN);  
  a_V_m_FLUT_STIN.increaseSizeTo(SHD.a_nodeN);  
  a_V_m_MYSA_2.increaseSizeTo(SHD.a_nodeN);  
  a_I_Naf.increaseSizeTo(SHD.a_nodeN);  
  a_I_Nap.increaseSizeTo(SHD.a_nodeN);  
  a_I_Ks.increaseSizeTo(SHD.a_nodeN);  
  a_I_leak.increaseSizeTo(SHD.a_nodeN);  
  a_m_Naf.increaseSizeTo(SHD.a_nodeN);  
  a_h_Naf.increaseSizeTo(SHD.a_nodeN);  
  a_p_Nap.increaseSizeTo(SHD.a_nodeN);  
  a_s_Ks.increaseSizeTo(SHD.a_nodeN);
  for (int i = 0; i < SHD.a_nodeN; ++i)
    {
      // Membrane potential for both node and internode segments
      a_V_m_node[i] = SHD.V_rest;
      a_V_m_MYSA_1[i] = SHD.V_rest;
      a_V_m_FLUT_STIN[i] = SHD.V_rest;
      a_V_m_MYSA_2[i] = SHD.V_rest;
      // Ionic currents and activation and inactivation variables for node segments
      // * Fast sodium current
      a = (6.57*(a_V_m_node[i]+10.4))/(1.0-exp(-(a_V_m_node[i]+10.4)/10.3));
      b = (0.304*(-(a_V_m_node[i]+15.7)))/(1.0-exp((a_V_m_node[i]+15.7)/9.16));
      a_m_Naf[i] = a / (a + b);
      a = (0.34*(-(a_V_m_node[i]+104.0)))/(1.0-exp((a_V_m_node[i]+104.0)/11.0));
      b = 12.6/(1.0+exp(-(a_V_m_node[i]+21.8)/13.4));
      a_h_Naf[i] = a / (a + b);
      // * Persistent sodium current
      a = (0.0353*(a_V_m_node[i]+17.0))/(1.0-exp(-(a_V_m_node[i]+17.0)/10.2));
      b = (0.000883*(-(a_V_m_node[i]+24.0)))/(1.0-exp((a_V_m_node[i]+24.0)/10.0));
      a_p_Nap[i] = a / (a + b);
      // * Slow potassium current
      a = 0.3/(1.0+exp((a_V_m_node[i]+43.0)/-5.0));
      b = 0.03/(1.0+exp((a_V_m_node[i]+80.0)/-1.0));
      a_s_Ks[i] = a / (a + b);
    }
  // Membrane potential for last node segment
  a_V_m_node_last = SHD.V_rest;
  // Ionic currents and activation and inactivation variables for last node segment
  // * Fast sodium current
  a = (6.57*(a_V_m_node_last+10.4))/(1.0-exp(-(a_V_m_node_last+10.4)/10.3));
  b = (0.304*(-(a_V_m_node_last+15.7)))/(1.0-exp((a_V_m_node_last+15.7)/9.16));
  a_m_Naf_last = a / (a + b);
  a = (0.34*(-(a_V_m_node_last+104.0)))/(1.0-exp((a_V_m_node_last+104.0)/11.0));
  b = 12.6/(1.0+exp(-(a_V_m_node_last+21.8)/13.4));
  a_h_Naf_last = a / (a + b);
  // * Persistent sodium current
  a = (0.0353*(a_V_m_node_last+17.0))/(1.0-exp(-(a_V_m_node_last+17.0)/10.2));
  b = (0.000883*(-(a_V_m_node_last+24.0)))/(1.0-exp((a_V_m_node_last+24.0)/10.0));
  a_p_Nap_last = a / (a + b);
  // * Slow potassium current
  a = 0.3/(1.0+exp((a_V_m_node_last+43.0)/-5.0));
  b = 0.03/(1.0+exp((a_V_m_node_last+80.0)/-1.0));
  a_s_Ks_last = a / (a + b);
  */
}

void MotoneuronUnit::update(RNG& rng)
{
  // ##### INPUT #####
  // Dendrite
  if (SHD.op_d_ramp)
    {
      if  (ITER <= (unsigned) (SHD.rampMiddle / SHD.deltaT))
        d_I_in += SHD.rampMax / (SHD.rampMiddle / SHD.deltaT);
      else
        d_I_in -= SHD.rampMax / (SHD.rampMiddle / SHD.deltaT);
    }
  // Soma
  if (SHD.op_s_ramp)
    {
      if  (ITER <= (unsigned) (SHD.rampMiddle / SHD.deltaT))
        s_I_in += SHD.rampMax / (SHD.rampMiddle / SHD.deltaT);
      else
        s_I_in -= SHD.rampMax / (SHD.rampMiddle / SHD.deltaT);
    }

  
  // Equations from (McIntyre and Grill, J. Neurophysiol., 2002)
  // ##### DENDRITE #####
  // Distal - [1]
  // Leak
  d_I_leak[1] = SHD.d_g_leak * (d_V_m[1] - SHD.dsi_E_leak);  
  // Axial current
  double avgDiameter =
    ( (ind_d_diameter[2]*(ind_d_length[2]/2.0)) + (ind_d_diameter[1]*(ind_d_length[1]/2.0)) )
    /
    ( (ind_d_length[2]/2.0) + (ind_d_length[1]/2.0) );
  double d_I_mddd = (d_V_m[1] - d_V_m[0]) /
    (
     (100.0 * ((ind_d_length[2]/2.0) + (ind_d_length[1]/2.0))) // Ra * (length/2)
     / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
     );
  // V_m
  d_V_m[1] -= (
               ((d_I_leak[1] + d_I_mddd) / SHD.Cm)
               + (d_I_in / (SHD.Cm * (ind_d_surfaceArea[2])))
               ) * SHD.deltaT;

  // Medial - [0]
  // Leak
  d_I_leak[0] = SHD.d_g_leak * (d_V_m[0] - SHD.dsi_E_leak);  
  // Axial current
  avgDiameter =
    ( (ind_d_diameter[1]*(ind_d_length[1]/2.0)) + (ind_d_diameter[0]*(ind_d_length[0]/2.0)) )
    /
    ( (ind_d_length[1]/2.0) + (ind_d_length[0]/2.0) );  
  double d_I_pdmd = (d_V_m[0] - d_V_m_last) /
    (
     (100.0 * ((ind_d_length[1]/2.0) + (ind_d_length[0]/2.0))) // Ra * (length/2)
     / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
     );
  avgDiameter =
    ( (ind_d_diameter[1]*(ind_d_length[1]/2.0)) + (ind_d_diameter[2]*(ind_d_length[2]/2.0)) )
    /
    ( (ind_d_length[1]/2.0) + (ind_d_length[2]/2.0) );  
  double d_I_ddmd = (d_V_m[0] - d_V_m[1]) /
    (
     (100.0 * ((ind_d_length[1]/2.0) + (ind_d_length[2]/2.0))) // Ra * (length/2)
     / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
     );  
  // V_m
  d_V_m[0] -= (
               ((d_I_leak[0] + d_I_pdmd + d_I_ddmd) / SHD.Cm)
               ) * SHD.deltaT;  

  // Proximal - _last
  // Leak
  d_I_leak_last = SHD.d_g_leak * (d_V_m_last - SHD.dsi_E_leak);  
  // Axial current
  avgDiameter =
    ( (ind_d_diameter[0]*(ind_d_length[0]/2.0)) + (ind_d_diameter[1]*(ind_d_length[1]/2.0)) )
    /
    ( (ind_d_length[0]/2.0) + (ind_d_length[1]/2.0) );
  double d_I_mdpd = (d_V_m_last - d_V_m[0]) /
    (
     (100.0 * ((ind_d_length[0]/2.0) + (ind_d_length[1]/2.0))) // Ra * (length/2)
     / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
     );
  double d_I_sdd = (d_V_m_last - s_V_m) /
    (
     (100.0 * (ind_d_length[0]/2.0) ) // Ra * (length/2)
     / (M_PI * pow(ind_d_diameter[0],2)) // pi * (diameter/2)^2
     );
  // V_m
  d_V_m_last -= (
                 ((d_I_leak_last + d_I_mdpd + d_I_sdd) / SHD.Cm)
                 ) * SHD.deltaT;

  
  
  // ##### SOMA #####
  // Fast sodium current
  s_m_Naf += (
              ( ( (0.4*(-(s_V_m+66.0)))/(exp(-(s_V_m+66.0)/5.0)-1.0) ) * (1.0-s_m_Naf) ) // alpha_m(1-m)
              - ( ( (0.4*(s_V_m+32.0))/(exp((s_V_m+32.0)/5.0)-1.0) ) * s_m_Naf ) // - beta_m*m
              ) * SHD.deltaT;
  s_h_Naf += (
              ( (1.0/(1.0+exp((s_V_m+65.0)/7.0))) - s_h_Naf ) // (s_h_inf - h)
              / ( 30.0/(exp((s_V_m+60.0)/15.0) + exp(-(s_V_m+60.0)/16.0)) ) // / tau_h
              ) * SHD.deltaT;
  s_I_Naf = ind_s_g_Naf * pow(s_m_Naf,3) * s_h_Naf * (s_V_m - SHD.dsi_E_Na);
  // Delayed rectifier potassium current
  s_n_Kdr += (
              ( (1.0/(1.0+exp((s_V_m+38.0)/-15.0))) - s_n_Kdr ) // (s_n_inf - n)
              / ( 5.0/(exp((s_V_m+50.0)/40.0)+exp(-(s_V_m+50.0)/50.0)) ) // / tau_n
              ) * SHD.deltaT;
  s_I_Kdr = ind_s_g_Kdr * pow(s_n_Kdr,4) * (s_V_m - SHD.dsi_E_K);
  // Calcium dynamics
  s_Ca_i += (
             0.01*(-(s_I_CaN+s_I_CaL)-(4.0*s_Ca_i))
             ) * SHD.deltaT;
  s_E_Ca = ((1000.0*R*309.15)/(2.0*F))*log(2.0/s_Ca_i); // s_Ca_o = 2.0
  // N-type calcium current
  s_m_CaN += (
              ( (1.0/(1.0+exp((s_V_m+32.0)/-5.0))) - s_m_CaN ) / 15.0 // (s_m_inf - s_m_CaN) / tau_m
              ) * SHD.deltaT;
  s_h_CaN += (
              ( (1.0/(1.0+exp((s_V_m+50.0)/5.0))) - s_h_CaN ) / 50.0 // (s_h_inf - s_h_CaN) / tau_h
              ) * SHD.deltaT;
  s_I_CaN = ind_s_g_CaN * pow(s_m_CaN,2) * s_h_CaN * (s_V_m - s_E_Ca);
  // L-type calcium current
  s_p_CaL += (
              ( (1.0/(1.0+exp((s_V_m+55.8)/-3.7))) - s_p_CaL ) / 400.0 // (s_p_inf - s_p_CaL) / tau_p
              ) * SHD.deltaT;
  s_I_CaL = ind_s_g_CaL * s_p_CaL * (s_V_m - s_E_Ca);
  // Calcium-activated potassium current
  s_I_KCa = ind_s_g_KCa * (pow(s_Ca_i,2)/(pow(s_Ca_i,2) + 0.000196)) * (s_V_m - SHD.dsi_E_K); // N.B. 0.000196 = 0.014^2
  // Leak
  s_I_leak = SHD.s_g_leak * (s_V_m - SHD.dsi_E_leak);
  // Axial current
  s_I_ds = (s_V_m - d_V_m_last) /
    (
     (100.0 * (ind_d_length[0]/2.0)) // Ra * (length/2.0)
     / (M_PI * pow(ind_d_diameter[0]/2.0,2)) // pi * (diameter/2)^2
     );
  s_I_is = (s_V_m - i_V_m) /
    (
     (100.0 * (ind_i_length/2.0)) // Ra * (length / 2)
     / (M_PI * pow(ind_i_diameter/2.0,2)) // pi * (diameter/2)^2
     );
  // V_m
  s_V_m -= (
            ((s_I_Naf + s_I_Kdr + s_I_CaN + s_I_CaL + s_I_KCa + s_I_leak +
              s_I_ds + s_I_is) / SHD.Cm)
            + (s_I_in / (SHD.Cm * ind_s_surfaceArea))
            ) * SHD.deltaT;

  

  // ##### IAS #####
  // Fast sodium current
  i_m_Naf += (
              ( ( (0.4*(-(i_V_m+60.0)))/(exp(-(i_V_m+60.0)/5.0)-1.0) ) * (1.0-i_m_Naf) ) // alpha_m(1-m)
              - ( ( (0.4*(i_V_m+40.0))/(exp((i_V_m+40.0)/5.0)-1.0) ) * i_m_Naf ) // - beta_m*m
              ) * SHD.deltaT;
  i_h_Naf += (
              ( (1.0/(1.0+exp((i_V_m+65.0)/7.0))) - i_h_Naf ) // (i_h_inf - h)
              / ( 30.0/(exp((i_V_m+60.0)/15.0) + exp(-(i_V_m+60.0)/16.0)) ) // / tau_h
              ) * SHD.deltaT;
  i_I_Naf = ind_i_g_Naf * pow(i_m_Naf,3) * i_h_Naf * (i_V_m - SHD.dsi_E_Na);
  // Persistent sodium current
  i_p_Nap += (
              ( ( (0.0353*(i_V_m+28.4))/(1.0-exp(-(i_V_m+28.4)/5.0)) ) * (1.0-i_p_Nap) ) // alpha_p(1-p)
              - ( ( (0.000883*(-(i_V_m+32.7)))/(1.0-exp((i_V_m+32.7)/5.0)) ) * i_p_Nap ) // - beta_p*p
              ) * SHD.deltaT;
  i_I_Nap = ind_i_g_Nap * pow(i_p_Nap,3) * (i_V_m - SHD.dsi_E_Na);
  // Delayed rectifier potassium current
  i_n_Kdr += (
              ( (1.0/(1.0+exp((i_V_m+38.0)/-15.0))) - i_n_Kdr ) // (i_n_inf - n)
              / ( 5.0/(exp((i_V_m+50.0)/40.0)+exp(-(i_V_m+50.0)/50.0)) ) // / tau_n
              ) * SHD.deltaT;
  i_I_Kdr = ind_i_g_Kdr * pow(i_n_Kdr,4) * (i_V_m - SHD.dsi_E_K);
  // Leak
  i_I_leak = SHD.i_g_leak * (i_V_m - SHD.dsi_E_leak);
  // Axial current
  i_I_si = (i_V_m - s_V_m) /
    (
     (100.0 * (ind_i_length/2.0)) // Ra * (length / 2)
     / (M_PI * pow(ind_i_diameter/2.0,2)) // pi * (diameter/2)^2
     );
  /*
  avgDiameter =
    ( (ind_a_node_diameter*(ind_a_node_length/2.0)) + (ind_i_diameter*(ind_i_length/2.0)) )
    /
    ( (ind_a_node_length/2.0) + (ind_i_length/2.0) );
  i_I_ai = (i_V_m - a_V_m_node[0]) / // is only every connected to first node
    (
     (35.0 * ((ind_a_node_length/2.0) + (ind_i_length/2.0))) // Ra * (length / 2)
     / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
     );
  */
  i_I_ai = 0.0;
  // V_m
  i_V_m -= (
            ((i_I_Naf + i_I_Nap + i_I_Kdr + i_I_leak + i_I_si + i_I_ai) / SHD.Cm)
            ) * SHD.deltaT;



  /*
  // ##### AXON #####
  // 1. First move along the axon
  for (int i = 0; i < SHD.a_nodeN; ++i)
    {
      // 1.1 The node
      // Fast sodium current
      a_m_Naf[i] += (
                     ( ( (6.57*(a_V_m_node[i]+10.4))/(1.0-exp(-(a_V_m_node[i]+10.4)/10.3)) ) * (1.0-a_m_Naf[i]) ) // alpha_m(1-m)
                     - ( ( (0.304*(-(a_V_m_node[i]+15.7)))/(1.0-exp((a_V_m_node[i]+15.7)/9.16)) ) * a_m_Naf[i] ) // - beta_m*m
                     ) * SHD.deltaT;
      a_h_Naf[i] += (
                     ( ( (0.34*(-(a_V_m_node[i]+104.0)))/(1.0-exp((a_V_m_node[i]+104.0)/11.0)) ) * (1.0-a_h_Naf[i]) ) // alpha_h(1-h)
                     - ( ( (12.6/(1.0+exp(-(a_V_m_node[i]+21.8)/13.4))) ) * a_h_Naf[i] ) // - beta_h*h
                     ) * SHD.deltaT;
      a_I_Naf[i] = ind_a_g_Naf * pow(a_m_Naf[i], 3) * a_h_Naf[i] * (a_V_m_node[i] - SHD.a_E_Na);
      // Persistent sodium current
      a_p_Nap[i] += (
                     ( ( (0.0353*(a_V_m_node[i]+17.0))/(1.0-exp(-(a_V_m_node[i]+17.0)/10.2)) ) * (1.0-a_p_Nap[i]) ) // alpha_p(1-p)
                     - ( ( (0.000883*(-(a_V_m_node[i]+24.0)))/(1.0-exp((a_V_m_node[i]+24.0)/10.0)) ) * a_p_Nap[i] ) // - beta_p*p
                     ) * SHD.deltaT;
      a_I_Nap[i] = ind_a_g_Nap * pow(a_p_Nap[i],3) * (a_V_m_node[i] - SHD.a_E_Na);
      // Slow potassium current
      a_s_Ks[i] += (
                    ( ( 0.3/(1.0+exp((a_V_m_node[i]+43.0)/-5.0)) ) * (1.0-a_s_Ks[i]) ) // alpha_s(1-p)
                    - ( ( 0.03/(1.0+exp((a_V_m_node[i]+80.0)/-1.0)) ) * a_s_Ks[i] ) // - beta_s*p
                    ) * SHD.deltaT;
      a_I_Ks[i] = ind_a_g_Ks * a_s_Ks[i] * (a_V_m_node[i] - SHD.a_E_K);
      // Leak
      a_I_leak[i] = SHD.a_g_leak * (a_V_m_node[i] - SHD.a_node_E_leak);
      // Axial current
      double a_I_prev_node = 0.0;
      double avgDiameter = 0.0;
      if (i == 0) // first node so connect to the IAS
        {
          avgDiameter =
            ( (ind_i_diameter*(ind_i_length/2.0)) + (ind_a_node_diameter*(ind_a_node_length/2.0)) )
            /
            ( (ind_i_length/2.0) + (ind_a_node_length/2.0) );
          a_I_prev_node = (a_V_m_node[0] - i_V_m) /
            (
             (35.0 * ((ind_i_length/2.0) + (ind_a_node_length/2.0))) // Ra * (length / 2)
             / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
             );
        }
      else // not first node so connect to previous MYSA_2
        {
          avgDiameter =
            ( (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) + (ind_a_node_diameter*(ind_a_node_length/2.0)) )
            /
            ( (ind_a_MYSA_length/2.0) + (ind_a_node_length/2.0) );
          a_I_prev_node = (a_V_m_node[i] - a_V_m_MYSA_2[i-1]) /
            (
             (35.0 * ((ind_a_MYSA_length/2.0) + (ind_a_node_length/2.0))) // Ra * (length / 2)
             / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
             );          
        }
      double a_I_MYSA_1_node = 0.0;
      avgDiameter =
        ( (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) + (ind_a_node_diameter*(ind_a_node_length/2.0)) )
        /
        ( (ind_a_MYSA_length/2.0) + (ind_a_node_length/2.0) );
      a_I_MYSA_1_node = (a_V_m_node[i] - a_V_m_MYSA_1[i]) /
        (
         (35.0 * ((ind_a_MYSA_length/2.0) + (ind_a_node_length/2.0))) // Ra * (length / 2)
         / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
         );                
      // V_m
      a_V_m_node[i] -= (
                        ((a_I_Naf[i] + a_I_Nap[i] + a_I_Ks[i] + a_I_leak[i] +
                          a_I_prev_node + a_I_MYSA_1_node) / SHD.Cm)
                        ) * SHD.deltaT;
      
      // 1.2 MYSA 1
      // Leak
      double a_I_leak_MYSA_1 = ind_a_g_MYSA * (a_V_m_MYSA_1[i] - SHD.a_internode_E_leak);
      // Axial current
      avgDiameter =
        ( (ind_a_node_diameter*(ind_a_node_length/2.0)) + (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) )
        /
        ( (ind_a_node_length/2.0) + (ind_a_MYSA_length/2.0) );
      double a_I_node_MYSA_1 = (a_V_m_MYSA_1[i] - a_V_m_node[i]) /
        (
         (35.0 * ((ind_a_node_length/2.0) + (ind_a_MYSA_length/2.0))) // Ra * (length / 2)
         / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
         );
      avgDiameter =
        ( (ind_a_FLUT_STIN_diameter*(ind_a_FLUT_STIN_length[i]/2.0)) + (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) )
        /
        ( (ind_a_FLUT_STIN_length[i]/2.0) + (ind_a_MYSA_length/2.0) );
      double a_I_FLUT_STIN_MYSA_1 = (a_V_m_MYSA_1[i] - a_V_m_FLUT_STIN[i]) /
        (
         (35.0 * ((ind_a_FLUT_STIN_length[i]/2.0) + (ind_a_MYSA_length/2.0))) // Ra * (length / 2)
         / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
         );                            
      // V_m
      a_V_m_MYSA_1[i] -= (
                          ((a_I_leak_MYSA_1 + a_I_node_MYSA_1 + a_I_FLUT_STIN_MYSA_1) / SHD.Cm)
                          ) * SHD.deltaT;

      // 1.2 FLUT & STIN
      // Leak
      double a_I_leak_FLUT_STIN = ind_a_g_FLUT_STIN * (a_V_m_FLUT_STIN[i] - SHD.a_internode_E_leak);
      // Axial current
      avgDiameter =
        ( (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) + (ind_a_FLUT_STIN_diameter*(ind_a_FLUT_STIN_length[i]/2.0)) )
        /
        ( (ind_a_MYSA_length/2.0) + (ind_a_FLUT_STIN_length[i]/2.0) );
      double a_I_MYSA_1_FLUT_STIN = (a_V_m_FLUT_STIN[i] - a_V_m_MYSA_1[i]) /
        (
         (35.0 * ((ind_a_MYSA_length/2.0) + (ind_a_FLUT_STIN_length[i]/2.0))) // Ra * (length / 2)
         / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
         );
      avgDiameter =
        ( (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) + (ind_a_FLUT_STIN_diameter*(ind_a_FLUT_STIN_length[i]/2.0)) )
        /
        ( (ind_a_MYSA_length/2.0) + (ind_a_FLUT_STIN_length[i]/2.0) );
      double a_I_MYSA_2_FLUT_STIN = (a_V_m_FLUT_STIN[i] - a_V_m_MYSA_2[i]) /
        (
         (35.0 * ((ind_a_MYSA_length/2.0) + (ind_a_FLUT_STIN_length[i]/2.0))) // Ra * (length / 2)
         / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
         );      
      // V_m
      a_V_m_FLUT_STIN[i] -= (
                             ((a_I_leak_FLUT_STIN + a_I_MYSA_1_FLUT_STIN + a_I_MYSA_2_FLUT_STIN) / SHD.Cm)
                             ) * SHD.deltaT;

      // 1.2 MYSA 2
      // Leak
      double a_I_leak_MYSA_2 = ind_a_g_MYSA * (a_V_m_MYSA_2[i] - SHD.a_internode_E_leak);
      // Axial current
      avgDiameter =
        ( (ind_a_FLUT_STIN_diameter*(ind_a_FLUT_STIN_length[i]/2.0)) + (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) )
        /
        ( (ind_a_FLUT_STIN_length[i]/2.0) + (ind_a_MYSA_length/2.0) );
      double a_I_FLUT_STIN_MYSA_2 = (a_V_m_MYSA_2[i] - a_V_m_FLUT_STIN[i]) /
        (
         (35.0 * ((ind_a_FLUT_STIN_length[i]/2.0) + (ind_a_MYSA_length/2.0))) // Ra * (length / 2)
         / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
         );
      double a_I_next_MYSA_2 = 0.0;
      if (i == SHD.a_nodeN-1) // last MYSA_2 so connect to node_last
        {
          avgDiameter =
            ( (ind_a_node_diameter*(ind_a_node_length/2.0)) + (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) )
            /
            ( (ind_a_node_length/2.0) + (ind_a_MYSA_length/2.0) );
          a_I_next_MYSA_2 = (a_V_m_MYSA_2[i] - a_V_m_node_last) /
            (
             (35.0 * ((ind_a_node_length/2.0) + (ind_a_MYSA_length/2.0))) // Ra * (length / 2)
             / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
             );
        }
      else // not last MYSA_2 so connect to next node
        {
          avgDiameter =
            ( (ind_a_node_diameter*(ind_a_node_length/2.0)) + (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) )
            /
            ( (ind_a_node_length/2.0) + (ind_a_MYSA_length/2.0) );
          a_I_next_MYSA_2 = (a_V_m_MYSA_2[i] - a_V_m_node[i+1]) /
            (
             (35.0 * ((ind_a_node_length/2.0) + (ind_a_MYSA_length/2.0))) // Ra * (length / 2)
             / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
             );          
        }
      // V_m
      a_V_m_MYSA_2[i] -= (
                          ((a_I_leak_MYSA_2 + a_I_FLUT_STIN_MYSA_2 + a_I_next_MYSA_2) / SHD.Cm)
                          ) * SHD.deltaT;      
    }
  
  // 2. The last node
  // Fast sodium current
  a_m_Naf_last += (
                   ( ( (6.57*(a_V_m_node_last+10.4))/(1.0-exp(-(a_V_m_node_last+10.4)/10.3)) ) * (1.0-a_m_Naf_last) ) // alpha_m(1-m)
                   - ( ( (0.304*(-(a_V_m_node_last+15.7)))/(1.0-exp((a_V_m_node_last+15.7)/9.16)) ) * a_m_Naf_last ) // - beta_m*m
                   ) * SHD.deltaT;
  a_h_Naf_last += (
                   ( ( (0.34*(-(a_V_m_node_last+104.0)))/(1.0-exp((a_V_m_node_last+104.0)/11.0)) ) * (1.0-a_h_Naf_last) ) // alpha_h(1-h)
                   - ( ( (12.6/(1.0+exp(-(a_V_m_node_last+21.8)/13.4))) ) * a_h_Naf_last ) // - beta_h*h
                   ) * SHD.deltaT;
  a_I_Naf_last = ind_a_g_Naf * pow(a_m_Naf_last, 3) * a_h_Naf_last * (a_V_m_node_last - SHD.a_E_Na);
  // Persistent sodium current
  a_p_Nap_last += (
                   ( ( (0.0353*(a_V_m_node_last+17.0))/(1.0-exp(-(a_V_m_node_last+17.0)/10.2)) ) * (1.0-a_p_Nap_last) ) // alpha_p(1-p)
                   - ( ( (0.000883*(-(a_V_m_node_last+24.0)))/(1.0-exp((a_V_m_node_last+24.0)/10.0)) ) * a_p_Nap_last ) // - beta_p*p
                   ) * SHD.deltaT;
  a_I_Nap_last = ind_a_g_Nap * pow(a_p_Nap_last,3) * (a_V_m_node_last - SHD.a_E_Na);
  // Slow potassium current
  a_s_Ks_last += (
                  ( ( 0.3/(1.0+exp((a_V_m_node_last+43.0)/-5.0)) ) * (1.0-a_s_Ks_last) ) // alpha_s(1-p)
                  - ( ( 0.03/(1.0+exp((a_V_m_node_last+80.0)/-1.0)) ) * a_s_Ks_last ) // - beta_s*p
                  ) * SHD.deltaT;
  a_I_Ks_last = ind_a_g_Ks * a_s_Ks_last * (a_V_m_node_last - SHD.a_E_K);
  // Leak
  a_I_leak_last = SHD.a_g_leak * (a_V_m_node_last - SHD.a_node_E_leak);
  // Axial current
  avgDiameter =
    ( (ind_a_MYSA_diameter*(ind_a_MYSA_length/2.0)) + (ind_a_node_diameter*(ind_a_node_length/2.0)) )
    /
    ( (ind_a_MYSA_length/2.0) + (ind_a_node_length/2.0) );
  double a_I_MYSA_2_node = (a_V_m_node_last - a_V_m_MYSA_2[SHD.a_nodeN-1]) /
    (
     (35.0 * ((ind_a_MYSA_length/2.0) + (ind_a_node_length/2.0))) // Ra * (length / 2)
     / (M_PI * pow(avgDiameter,2)) // pi * (diameter/2)^2
     );
  // V_m
  a_V_m_node_last -= (
                      ((a_I_Naf_last + a_I_Nap_last + a_I_Ks_last + a_I_leak_last +
                        a_I_MYSA_2_node) / SHD.Cm)
                      ) * SHD.deltaT;
  */
}

MotoneuronUnit::~MotoneuronUnit()
{
}

