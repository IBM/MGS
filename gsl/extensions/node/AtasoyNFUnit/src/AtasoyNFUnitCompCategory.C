// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "AtasoyNFUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_AtasoyNFUnitCompCategory.h"
#include <fstream>
#include <sstream>
#include <iostream>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()

AtasoyNFUnitCompCategory::AtasoyNFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_AtasoyNFUnitCompCategory(sim, modelName, ndpList)
{
}

void AtasoyNFUnitCompCategory::initializeShared(RNG& rng) 
{
  //read max degree of laplacian to determine diffusion time constants
  //const char* fname = SHD.laplacian_filename.c_str();
  std::ifstream ifs("L.txt", std::ifstream::in); 
  std::string line;
  std::getline(ifs, line);
  std::getline(ifs, line);
  int maxRow, maxCol;
  double maxDegree;
  std::stringstream line_stream(line);
  line_stream >> maxRow >> maxCol >> maxDegree; 
  ifs.close();

  SHD.t_diff_max = std::round(maxDegree);
  SHD.tau = 1.0/maxDegree;
  SHD.t_diffE2E = SHD.sigma_E2E * maxDegree;
  SHD.t_diffE2I = SHD.sigma_E2I * maxDegree;
  SHD.t_diffI2E = SHD.sigma_I2E * maxDegree;
  SHD.t_diffI2I = SHD.sigma_I2I * maxDegree;

  if(!SHD.lineOffset) SHD.lineOffset=4;
  
}

