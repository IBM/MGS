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
#include "TraubIAFUnitDataCollector.h"
#include "CG_TraubIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void TraubIAFUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, 
                             std::pair< 
                               std::pair<double*, double*>, // second.first.first, second.first.second
                               std::pair<
                                 std::pair<bool*, float*>, // second.second.first.first, second.second.first.second
                                 double* // second.second.second
                                 >
                               >
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==voltages.size());
  assert(slices.size()==thresholds.size());
  assert(slices.size()==spikes.size());
  assert(slices.size()==spikevoltages.size());
  assert(slices.size()==LFP_synapses.size());
  int sz=voltages.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(
                                                        std::make_pair(voltages[j], thresholds[j]),
                                                        std::make_pair(
                                                                       std::make_pair(spikes[j], spikevoltages[j]),
                                                                       LFP_synapses[j]
                                                                       )
                                                        );
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
    if (mxslice<slices[j]) mxslice=slices[j];
  }
  voltages.clear();
  thresholds.clear();
  spikes.clear();
  spikevoltages.clear();
  LFP_synapses.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, 
                             std::pair< 
                               std::pair<double*, double*>,
                               std::pair<
                                 std::pair<bool*, float*>,
                                 double*
                                 >
                               >
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, 
             std::map<unsigned, 
                      std::pair< 
                        std::pair<double*, double*>,
                        std::pair<
                          std::pair<bool*, float*>,
                            double*
                          >
                        >
                      >
             >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      std::map<unsigned, 
               std::pair< 
                 std::pair<double*, double*>,
                 std::pair<
                   std::pair<bool*, float*>,
                     double*
                   >
                 >
               >::iterator miter3, mend3=miter2->second.end();
      for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3) {
        voltages.push_back(miter3->second.first.first);
        thresholds.push_back(miter3->second.first.second);
        spikes.push_back(miter3->second.second.first.first);
        spikevoltages.push_back(miter3->second.second.first.second);
        LFP_synapses.push_back(miter3->second.second.second);
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
  
  std::ostringstream os_voltage, os_threshold, os_spike, os_spikevoltage,
    os_LFP_synapses;

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

  if (op_saveThresholds)
    {  
      os_threshold<<directory<<"Threshold"<<fileExt;
      threshold_file=new std::ofstream(os_threshold.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      threshold_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      threshold_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      threshold_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
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
  
  if (op_saveSpikeVoltages)
    {  
      os_spikevoltage<<directory<<"SpikeVoltage"<<fileExt;
      spikevoltage_file=new std::ofstream(os_spikevoltage.str().c_str(),
                                          std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      spikevoltage_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      spikevoltage_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      spikevoltage_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveLFPs)
    {  
      os_LFP_synapses<<directory<<"LFP_FSIsynapses"<<fileExt;
      LFP_synapses_file=new std::ofstream(os_LFP_synapses.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      LFP_synapses_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      LFP_synapses_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));  
      LFP_synapses_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void TraubIAFUnitDataCollector::finalize(RNG& rng) 
{
  if (op_saveVoltages)
    {
      voltage_file->close();
      delete voltage_file;
    }
  
  if (op_saveThresholds)
    {
      threshold_file->close();
      delete threshold_file;
    }
  
  if (op_saveSpikes)
    {
      spike_file->close();
      delete spike_file;
    }
  
  if (op_saveSpikeVoltages)
    {
      spikevoltage_file->close();
      delete spikevoltage_file;
    }
  
  if (op_saveLFPs)
    {
      LFP_synapses_file->close();
      delete LFP_synapses_file;  
    }
}


void TraubIAFUnitDataCollector::dataCollectionSpikes(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveSpikes)
    {
      if (!op_saveSpikesAll)
        {
          ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
          unsigned temp = getSimulation().getIteration();
          for (int n=0; iter!=end; ++iter, n++)
            {
              if (**iter)
                {
                  spike_file->write(reinterpret_cast<char *>(&n), sizeof(n));
                  spike_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
                }
            }
        }
      else 
        {
          ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
          ShallowArray<int>::iterator iterX=rows.begin(), endX=rows.end();      
          ShallowArray<int>::iterator iterY=cols.begin(), endY=cols.end();      
          ShallowArray<int>::iterator iterZ=slices.begin(), endZ=slices.end();      
          unsigned temp = getSimulation().getIteration();
          int x=0,y=0,z=0;
          for (int n=0; iter!=end; ++iter, ++iterX, ++iterY, ++iterZ, n++)
            {
              x = *iterX;
              y = *iterY;
              z = *iterZ;          
              if ((x >= Xmin) && (x <= Xmax)
                  && (y >= Ymin) && (y <= Ymax)
                  && (z >= Zmin) && (z <= Zmax))
                if (**iter)
                  {
                    spike_file->write(reinterpret_cast<char *>(&n), sizeof(n));
                    spike_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
                  }
            }
        }
    }
}

void TraubIAFUnitDataCollector::dataCollectionOther(Trigger* trigger, NDPairList* ndPairList) 
{
  ShallowArray<double*>::iterator iter, end;
  float temp = 0.;
  if (op_saveVoltages)
    {
      iter=voltages.begin();
      end=voltages.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          voltage_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_saveThresholds)
    {
      iter=thresholds.begin();
      end=thresholds.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          threshold_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  
  if (op_saveSpikeVoltages)
    {
      ShallowArray<float*>::iterator iter3=spikevoltages.begin(), end3=spikevoltages.end();
      for (int n=0; iter3!=end3; ++iter3)
        spikevoltage_file->write(reinterpret_cast<char *>(*iter3), sizeof(float));
    }

  if (op_saveLFPs)
    {  
      iter=LFP_synapses.begin();
      end=LFP_synapses.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          LFP_synapses_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
}

void TraubIAFUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_TraubIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_TraubIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

TraubIAFUnitDataCollector::TraubIAFUnitDataCollector() 
   : CG_TraubIAFUnitDataCollector()
{
}

TraubIAFUnitDataCollector::~TraubIAFUnitDataCollector() 
{
}

void TraubIAFUnitDataCollector::duplicate(std::auto_ptr<TraubIAFUnitDataCollector>& dup) const
{
   dup.reset(new TraubIAFUnitDataCollector(*this));
}

void TraubIAFUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new TraubIAFUnitDataCollector(*this));
}

void TraubIAFUnitDataCollector::duplicate(std::auto_ptr<CG_TraubIAFUnitDataCollector>& dup) const
{
   dup.reset(new TraubIAFUnitDataCollector(*this));
}

