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
#include "LeakyIAFUnitDataCollector.h"
#include "CG_LeakyIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void LeakyIAFUnitDataCollector::initialize(RNG& rng) 
{

  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, 
                             std::pair<float*, bool*> // second.first, second.second
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==voltages.size());
  assert(slices.size()==spikes.size());
  int sz=voltages.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(voltages[j], spikes[j]);
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
    if (mxslice<slices[j]) mxslice=slices[j];
  }
  voltages.clear();
  spikes.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, 
                             std::pair<float*, bool*>
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, 
             std::map<unsigned, 
                      std::pair<float*, bool*>
                      >
             >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      std::map<unsigned, 
               std::pair<float*, bool*>
               >::iterator miter3, mend3=miter2->second.end();
      for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3) {
        voltages.push_back(miter3->second.first);
        spikes.push_back(miter3->second.second);
      }
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
  
  std::ostringstream os_voltage, os_spike;
  
  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;  

  if (op_saveVoltages)
    {
      os_voltage<<directory<<"Voltage"<<fileExt;
      voltage_file=new std::ofstream(os_voltage.str().c_str(),
                                     std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      voltage_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      voltage_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      voltage_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
  
  if (op_saveSpikes)
    {
      os_spike<<directory<<"Spike"<<fileExt;
      spike_file=new std::ofstream(os_spike.str().c_str(),
                                   std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      spike_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      spike_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      spike_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void LeakyIAFUnitDataCollector::finalize(RNG& rng) 
{
  if (op_saveVoltages)
    {
      voltage_file->close();
      delete voltage_file;
    }
  
  if (op_saveSpikes)
    {
      spike_file->close();
      delete spike_file;
    }  
}

void LeakyIAFUnitDataCollector::dataCollectionSpikes(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveSpikes)
    {
      ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
      unsigned temp2 = getSimulation().getIteration();
      for (int n=0; iter!=end; ++iter, n++)
        if (**iter)
          {
            spike_file->write(reinterpret_cast<char *>(&n), sizeof(n));
            spike_file->write(reinterpret_cast<char *>(&temp2), sizeof(temp2));
          }
    }
}

void LeakyIAFUnitDataCollector::dataCollectionOther(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveVoltages)
    {
      ShallowArray<float*>::iterator iter=voltages.begin(), end=voltages.end();
      for (int n=0; iter!=end; ++iter)
        voltage_file->write(reinterpret_cast<char *>(*iter), sizeof(**iter));
    }  
}

void LeakyIAFUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LeakyIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_LeakyIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

LeakyIAFUnitDataCollector::LeakyIAFUnitDataCollector() 
  : CG_LeakyIAFUnitDataCollector()
{
}

LeakyIAFUnitDataCollector::~LeakyIAFUnitDataCollector() 
{
}

void LeakyIAFUnitDataCollector::duplicate(std::auto_ptr<LeakyIAFUnitDataCollector>& dup) const
{
  dup.reset(new LeakyIAFUnitDataCollector(*this));
}

void LeakyIAFUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new LeakyIAFUnitDataCollector(*this));
}

void LeakyIAFUnitDataCollector::duplicate(std::auto_ptr<CG_LeakyIAFUnitDataCollector>& dup) const
{
  dup.reset(new LeakyIAFUnitDataCollector(*this));
}

