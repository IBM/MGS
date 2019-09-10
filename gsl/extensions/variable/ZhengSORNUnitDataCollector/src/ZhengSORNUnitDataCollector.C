// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "ZhengSORNUnitDataCollector.h"
#include "CG_ZhengSORNUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <utility>

void ZhengSORNUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, std::map<unsigned, bool*> > sorter;
  assert(rows.size()==cols.size());
  assert(cols.size()==spikes.size());
  int sz=spikes.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=spikes[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  spikes.clear();
  std::map<unsigned, std::map<unsigned, bool*> >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, bool*>::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      spikes.push_back(miter2->second);
    }
  }

  // Create the output files...
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<directory.c_str()<<";";
  try {
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};

  std::ostringstream os_spikes;

  int Xdim = (int) mxrow+1;
  int Ydim = (int) mxcol+1;
  //int Zdim = (int) mxrow+1;    
  
  if (op_saveSpikes)
    {
      os_spikes<<directory<<fileName<<fileExt;
      if (binary)
        {          
          spikesFile = new std::ofstream(os_spikes.str().c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
          spikesFile->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
          spikesFile->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
          //X_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
        }
      else
        {
          spikesFile = new std::ofstream(os_spikes.str().c_str());
          std::ofstream& output = *spikesFile;
          output << mxrow+1 << " " << mxcol+1 << std::endl << std::endl;          
        }
    }
}

void ZhengSORNUnitDataCollector::finalize(RNG& rng) 
{
  spikesFile->close();
  delete spikesFile;
}

void ZhengSORNUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  /*double t= getSimulation().getIteration()*deltaT;
  std::ofstream& output=*file;
  int end=spikes.size();
  for (int idx=0; idx!=end; ++idx) {
    if (*(spikes[idx]))
      output << std::fixed << std::setprecision(3) << t <<" "<< idx << std::endl;
  }*/ 
  if (op_saveSpikes)
    {
      if (binary)
        {
          unsigned t = getSimulation().getIteration();
          ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
          for (int n=0; iter!=end; ++iter, n++)
            {
              // Only write if spiking
              if (**iter)
                {
                  spikesFile->write(reinterpret_cast<char *>(&n), sizeof(n));
                  spikesFile->write(reinterpret_cast<char *>(&t), sizeof(t));
                }
            }
        }
      else
        {
          std::ofstream& output = *spikesFile;
          double t = getSimulation().getIteration();
          int end=spikes.size();
          for (int idx=0; idx!=end; ++idx) {
            if (*(spikes[idx]))
              output << std::fixed << std::setprecision(3) << t <<" "<< idx << std::endl;
            }
        }
    }
}

void ZhengSORNUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ZhengSORNUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_ZhengSORNUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

ZhengSORNUnitDataCollector::ZhengSORNUnitDataCollector() 
   : CG_ZhengSORNUnitDataCollector()
{
}

ZhengSORNUnitDataCollector::~ZhengSORNUnitDataCollector() 
{
}

void ZhengSORNUnitDataCollector::duplicate(std::unique_ptr<ZhengSORNUnitDataCollector>& dup) const
{
   dup.reset(new ZhengSORNUnitDataCollector(*this));
}

void ZhengSORNUnitDataCollector::duplicate(std::unique_ptr<Variable>& dup) const
{
   dup.reset(new ZhengSORNUnitDataCollector(*this));
}

void ZhengSORNUnitDataCollector::duplicate(std::unique_ptr<CG_ZhengSORNUnitDataCollector>& dup) const
{
   dup.reset(new ZhengSORNUnitDataCollector(*this));
}

