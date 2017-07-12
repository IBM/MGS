// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "SpineIAFUnitCompCategory.h"
#include "NDPairList.h"
#include "CG_SpineIAFUnitCompCategory.h"
#include <fstream>
#include <sstream>

#define SHD getSharedMembers()

SpineIAFUnitCompCategory::SpineIAFUnitCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList)
  : CG_SpineIAFUnitCompCategory(sim, modelName, ndpList)
{
}

void SpineIAFUnitCompCategory::initializeShared(RNG& rng)
{
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<SHD.sharedDirectory.c_str()<<";";
  try {
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};
  int rank=getSimulation().getRank();
  int n=0;
  if (SHD.op_saveWeights)
    {
      // Take it in turn opening and creating the file to create the stream on each node
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          os_weights<<SHD.sharedDirectory<<"AMPAWeights"<<SHD.sharedFileExt;
          weights_file=new std::ofstream(os_weights.str().c_str(),
                                        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
          weights_file->close();
        }
        ++n;
        MPI::COMM_WORLD.Barrier(); // wait node creating the stream to finish
      }
      // Now take it in turn writing to the file, where rank 0 also clears the file.
      n=0;
      while (n<getSimulation().getNumProcesses()) {
        if (n==rank) {
          ShallowArray<SpineIAFUnit>::iterator it = _nodes.begin();
          ShallowArray<SpineIAFUnit>::iterator end = _nodes.end();
          weights_file->open(os_weights.str().c_str(),
                            std::ofstream::out | std::ofstream::app | std::ofstream::binary);
          for (; it != end; ++it)
            (*it).outputWeights(*weights_file);
          weights_file->close();
        }
        ++n;
        MPI::COMM_WORLD.Barrier(); // wait for node writing to finish
      }
    }
}

