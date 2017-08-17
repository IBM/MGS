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
#include "WaveDriverUnitDataCollector.h"
#include "CG_WaveDriverUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void WaveDriverUnitDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             double*   // second
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==waves.size());
  int sz=waves.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]][slices[j]]=waves[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
    if (mxslice<slices[j]) mxslice=slices[j];
  }
  waves.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             double*
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, 
             std::map<unsigned, 
                      double*
                      >
             >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      std::map<unsigned, 
               double*
               >::iterator miter3, mend3=miter2->second.end();
      for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3) {
        waves.push_back(miter3->second);
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
  
  std::ostringstream os_wave;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;  

  if (op_saveWaves)
    {
      os_wave<<directory<<"Wave"<<fileExt;
      wave_file=new std::ofstream(os_wave.str().c_str(),
                                  std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      wave_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      wave_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      wave_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void WaveDriverUnitDataCollector::finalize(RNG& rng) 
{
  if (op_saveWaves)
    {
      wave_file->close();
      delete wave_file;
    }  
}

void WaveDriverUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  ShallowArray<double*>::iterator iter, end;
  float temp = 0.;
  if (op_saveWaves)
    {
      iter=waves.begin();
      end=waves.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          wave_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }  
}

void WaveDriverUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_WaveDriverUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_WaveDriverUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

WaveDriverUnitDataCollector::WaveDriverUnitDataCollector() 
  : CG_WaveDriverUnitDataCollector()
{
}

WaveDriverUnitDataCollector::~WaveDriverUnitDataCollector() 
{
}

void WaveDriverUnitDataCollector::duplicate(std::auto_ptr<WaveDriverUnitDataCollector>& dup) const
{
  dup.reset(new WaveDriverUnitDataCollector(*this));
}

void WaveDriverUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new WaveDriverUnitDataCollector(*this));
}

void WaveDriverUnitDataCollector::duplicate(std::auto_ptr<CG_WaveDriverUnitDataCollector>& dup) const
{
  dup.reset(new WaveDriverUnitDataCollector(*this));
}

