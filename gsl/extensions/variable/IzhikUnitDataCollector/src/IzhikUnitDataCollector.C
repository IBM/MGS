// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#include "Mgs.h"
#include "IzhikUnitDataCollector.h"
#include "CG_IzhikUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>


//#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()
//#define TIME ITER*getSharedMembers().deltaT
#define RANK getSimulation().getRank()

void IzhikUnitDataCollector::initialize(RNG& rng) 
{



  {std::ostringstream os1;
  os1<<"Voltage_"<<fileName;
  voltage_file=new std::ofstream(os1.str().c_str(), std::ofstream::out | std::ofstream::app);}


  {std::ostringstream os1;
  os1<<"Spike_"<<fileName;
  spike_file=new std::ofstream(os1.str().c_str(), std::ofstream::out | std::ofstream::app);}

  {std::ostringstream os1;
  os1<<"InhibTrans_"<<fileName;
  trans_file=new std::ofstream(os1.str().c_str(), std::ofstream::out | std::ofstream::app);}




  std::map<unsigned, std::map<unsigned, std::pair<float*, std::pair<double*, bool* > > > >  sorter;

  assert(rows.size()==cols.size());
  assert(cols.size()==V.size());
  assert(cols.size()==spikes.size());
  assert(cols.size()==g.size());

  int sz=V.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=std::pair<float*, std::pair<double*, bool*> >(V[j], std::pair<double*, bool*>(g[j],spikes[j]));
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  V.clear();
  spikes.clear();
  g.clear();
  std::map<unsigned, std::map<unsigned, std::pair<float*, std::pair<double*, bool* > > > >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, std::pair<float*, std::pair<double*, bool* > > >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      V.push_back(miter2->second.first);
      spikes.push_back(miter2->second.second.second);
      g.push_back(miter2->second.second.first);
    }
  }





  

  //spike_output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;


  //file=new std::ofstream(fileName.c_str());
  //std::ofstream& output=*file;
  //output<<mxrow+1<<" "<<mxcol+1<<std::endl<<std::endl;




}

void IzhikUnitDataCollector::finalize(RNG& rng) 
{

  *spike_file<<std::endl;
  *voltage_file<<std::endl;
  *trans_file<<std::endl;

  spike_file->close();
  voltage_file->close();
  trans_file->close();
}

void IzhikUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{

  {std::ofstream& output=*voltage_file;
  output<<ITER*deltaT<<" ";//std::endl;
  ShallowArray<float*>::iterator iter=V.begin(), end=V.end();
  for (int col=0; iter!=end && col<maxoutnum; ++iter, ++col) {
    output<<**iter<<" ";
    /*
    if (++col>mxcol) {
      output<<std::endl;
      col=0;
    }
    */
  }
  output<<std::endl;}


  {std::ofstream& output=*trans_file;
  output<<ITER*deltaT<<" ";//std::endl;
  ShallowArray<double*>::iterator iter=g.begin(), end=g.end();
  for (int col=0; iter!=end && col<maxoutnum; ++iter, ++col) {
    output<<**iter<<" ";
    /*
    if (++col>mxcol) {
      output<<std::endl;
      col=0;
    }
    */
  }
  output<<std::endl;}




}

void IzhikUnitDataCollector::dataCollectionSpike(Trigger* trigger, NDPairList* ndPairList) 
{


  std::ofstream& output=*spike_file;
  //output<<getSimulation().getIteration()<<" ";//std::endl;
  ShallowArray<bool*>::iterator iter=spikes.begin(), end=spikes.end();
  for (int col=0; iter!=end; ++iter, ++col) {
    if (**iter == true) 
      output<<getSimulation().getIteration()<<" "<< col << std::endl;
    /*
    if (++col>mxcol) {
      output<<std::endl;
      col=0;
    }
    */
  }

  // output<<std::endl;

}




void IzhikUnitDataCollector::getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_IzhikUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_IzhikUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

IzhikUnitDataCollector::IzhikUnitDataCollector() 
   : CG_IzhikUnitDataCollector()
{
}

IzhikUnitDataCollector::~IzhikUnitDataCollector() 
{
  if (voltage_file) delete voltage_file; if (spike_file) delete spike_file;
  if (trans_file) delete trans_file;
}

void IzhikUnitDataCollector::duplicate(std::unique_ptr<IzhikUnitDataCollector>&& dup) const
{
   dup.reset(new IzhikUnitDataCollector(*this));
}

void IzhikUnitDataCollector::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new IzhikUnitDataCollector(*this));
}

void IzhikUnitDataCollector::duplicate(std::unique_ptr<CG_IzhikUnitDataCollector>&& dup) const
{
   dup.reset(new IzhikUnitDataCollector(*this));
}

