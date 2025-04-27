// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "MihalasNieburIAFUnitDataCollector.h"
#include "CG_MihalasNieburIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void MihalasNieburIAFUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, std::map<unsigned, std::pair< std::pair<double*, double*>, bool*> > > sorter;
  assert(rows.size()==cols.size());
  assert(cols.size()==voltages.size());
  assert(cols.size()==thresholds.size());
  assert(cols.size()==spikes.size());
  int sz=voltages.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=std::pair< std::pair<double*, double*>, bool*>( std::pair<double*, double*>(voltages[j], thresholds[j]), spikes[j]);
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  voltages.clear();
  thresholds.clear();
  spikes.clear();
  std::map<unsigned, std::map<unsigned, std::pair< std::pair<double*, double*>, bool*> > >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, std::pair< std::pair<double*, double*>, bool*> >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      voltages.push_back(miter2->second.first.first);
      thresholds.push_back(miter2->second.first.second);
      spikes.push_back(miter2->second.second);
    }
  }

  // Create the output files...
  std::ostringstream os_voltage, os_threshold, os_spike;
  os_voltage<<"Voltage_"<<fileName;
  voltage_file=new std::ofstream(os_voltage.str().c_str());
  std::ofstream& voltage_output=*voltage_file;
  voltage_output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;

  os_threshold<<"Threshold_"<<fileName;
  threshold_file=new std::ofstream(os_threshold.str().c_str());
  std::ofstream& threshold_output=*threshold_file;
  threshold_output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;

  os_spike<<"Spike_"<<fileName;
  spike_file=new std::ofstream(os_spike.str().c_str());
  std::ofstream& spike_output=*spike_file;
  spike_output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;
}

void MihalasNieburIAFUnitDataCollector::finalize(RNG& rng) 
{
  voltage_file->close();
  delete voltage_file;

  threshold_file->close();
  delete threshold_file;
}

void MihalasNieburIAFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  std::ofstream& voltage_output=*voltage_file;
  voltage_output<<getSimulation().getIteration()<<std::endl;
  ShallowArray<double*>::iterator iter=voltages.begin(), end=voltages.end();
  for (int col=0; iter!=end; ++iter) {
    voltage_output<<**iter<<" ";
    if (++col>mxcol) {
      voltage_output<<std::endl;
      col=0;
    }
  }
  voltage_output<<std::endl;

  std::ofstream& threshold_output=*threshold_file;
  threshold_output<<getSimulation().getIteration()<<std::endl;
  iter=thresholds.begin(), end=thresholds.end();
  for (int col=0; iter!=end; ++iter) {
    threshold_output<<**iter<<" ";
    if (++col>mxcol) {
      threshold_output<<std::endl;
      col=0;
    }
  }
  threshold_output<<std::endl;

  std::ofstream& spike_output=*spike_file;
  spike_output<<getSimulation().getIteration()<<std::endl;
  ShallowArray<bool*>::iterator iter2=spikes.begin(), end2=spikes.end();
  for (int col=0; iter2!=end2; ++iter2) {
    spike_output<<**iter2<<" ";
    if (++col>mxcol) {
      spike_output<<std::endl;
      col=0;
    }
  }
  spike_output<<std::endl;
}

void MihalasNieburIAFUnitDataCollector::getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MihalasNieburIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MihalasNieburIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

MihalasNieburIAFUnitDataCollector::MihalasNieburIAFUnitDataCollector() 
   : CG_MihalasNieburIAFUnitDataCollector()
{
}

MihalasNieburIAFUnitDataCollector::~MihalasNieburIAFUnitDataCollector() 
{
}

void MihalasNieburIAFUnitDataCollector::duplicate(std::unique_ptr<MihalasNieburIAFUnitDataCollector>&& dup) const
{
   dup.reset(new MihalasNieburIAFUnitDataCollector(*this));
}

void MihalasNieburIAFUnitDataCollector::duplicate(std::unique_ptr<Variable>duplicate(std::unique_ptr<Variable>& dup)duplicate(std::unique_ptr<Variable>& dup) dup) const
{
   dup.reset(new MihalasNieburIAFUnitDataCollector(*this));
}

void MihalasNieburIAFUnitDataCollector::duplicate(std::unique_ptr<CG_MihalasNieburIAFUnitDataCollector>&& dup) const
{
   dup.reset(new MihalasNieburIAFUnitDataCollector(*this));
}

