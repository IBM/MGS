// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "AtasoyNFUnit.h"
#include "CG_AtasoyNFUnit.h"
#include "rndm.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

void AtasoyNFUnit::initialize(RNG& rng) 
{
  // initial field conditions
  E = 0.5 + 0.2 * drandom(rng);
  I = 0.5 + 0.2 * drandom(rng);
  E_diff = E;
  I_diff = I;
}

void AtasoyNFUnit::diffusion(RNG& rng) 
{
  ShallowArray<DiffusionInput>::iterator it, end=D_in.end();
  double dEdt=0.0;
  double dIdt=0.0;
  for (it=D_in.begin(); it!=end; ++it) {
    dEdt += *(it->E_diff) * it->weight;
    dIdt += *(it->I_diff) * it->weight;
  }

  E_diff += SHD.tau * dEdt;
  I_diff += SHD.tau * dIdt;

  if (t_diff == SHD.t_diffE2E) {
    diff_E2E = E_diff;
  }
  if (t_diff == SHD.t_diffE2I) {
    diff_E2I = E_diff;
  }
  if (t_diff == SHD.t_diffI2E) {
    diff_I2E = I_diff;
  }
  if (t_diff == SHD.t_diffI2I) {
    diff_I2I = I_diff;
  }

  t_diff++;
}

void AtasoyNFUnit::reaction(RNG& rng) 
{
  if (t_diff==SHD.t_diff_max) {
    E += (-SHD.dE * E + 1.0 / (1.0 + exp(-(SHD.alpha_E2E * diff_E2E + SHD.alpha_I2E * diff_I2E)))) * SHD.dt;
    I += (-SHD.dI * I + 1.0 / (1.0 + exp(-(SHD.alpha_E2I * diff_E2I + SHD.alpha_I2I * diff_I2I)))) * SHD.dt;
    E_diff = E;
    I_diff = I;
    t_diff=0;
  }
}

bool AtasoyNFUnit::checkForConnection(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AtasoyNFUnitInAttrPSet* CG_inAttrPset, CG_AtasoyNFUnitOutAttrPSet* CG_outAttrPset) 
{
  bool rval=false;
  unsigned preIdx = CG_node->getNode()->getIndex();
  unsigned postIdx = getNode()->getIndex();
  //const char* fname = SHD.laplacian_filename.c_str();
  std::ifstream ifs("L.txt", std::ifstream::in);
  std::string line;
  
  // skip header
  for (int i=0; i<SHD.lineOffset; i++) {
    std::getline(ifs, line);
  }
  
  // import connectivity, set indices (-/+ 1 for compatibility w/ matlab)
  int post, pre;
  double val;
  while (std::getline(ifs, line)) {
    std::stringstream line_stream(line);
    line_stream >> post >> pre >> val;
    if(postIdx==(post-1) && preIdx==(pre-1)) {
      D_in[D_in.size()-1].weight = val;
      D_in[D_in.size()-1].row = getGlobalIndex()+1;  
      D_in[D_in.size()-1].col = CG_node->getGlobalIndex()+1;
      rval=true;
      break;
    }
  }
  return rval;
}   

AtasoyNFUnit::~AtasoyNFUnit() 
{
}

