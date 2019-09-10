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
#include "NazeSORNUnitDataCollectorSuppl.h"
#include "CG_NazeSORNUnitDataCollectorSuppl.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <utility>

void NazeSORNUnitDataCollectorSuppl::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, std::map<unsigned, double*> > sorter;
  assert(rows.size()==cols.size());
  assert(cols.size()==thresholds.size());
  int sz=thresholds.size();
  int mxrow=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]]=thresholds[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
  }
  thresholds.clear();
  std::map<unsigned, std::map<unsigned, double*> >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1) {
    std::map<unsigned, double*>::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      thresholds.push_back(miter2->second);
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

  std::ostringstream os_thresholds;

  int Xdim = (int) mxrow+1;
  int Ydim = (int) mxcol+1;
  //int Zdim = (int) mxrow+1;    
  
  if (op_saveThresholds)
    {
      os_thresholds<<directory<<fileName<<fileExt;
      if (binary)
        {          
          thresholdsFile = new std::ofstream(os_thresholds.str().c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
          thresholdsFile->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
          thresholdsFile->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
          //X_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
        }
      else
        {
          thresholdsFile = new std::ofstream(os_thresholds.str().c_str());
          std::ofstream& output = *thresholdsFile;
          output << mxrow+1 << " " << mxcol+1 << std::endl << std::endl;          
        }
    }
}

void NazeSORNUnitDataCollectorSuppl::finalize(RNG& rng) 
{
  thresholdsFile->close();
  delete thresholdsFile;
}

void NazeSORNUnitDataCollectorSuppl::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveThresholds)
  {
    if (binary)
      {
	//unsigned t = getSimulation().getIteration();
	ShallowArray<double*>::iterator iter=thresholds.begin(), end=thresholds.end();
	for (iter; iter!=end; iter++)
	  {
	    thresholdsFile->write(reinterpret_cast<char *>(*iter), sizeof(double));
	  }
      }
    else
      {
	std::ofstream& output = *thresholdsFile;
	double t = getSimulation().getIteration();
	int end=thresholds.size();
	output << std::fixed << std::setprecision(3) << t << std::endl;
	for (int idx=0; idx!=end; ++idx) {
	  output << thresholds[idx] << " " ;
	}
	output << std::endl;
      }
  }
}

void NazeSORNUnitDataCollectorSuppl::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_NazeSORNUnitDataCollectorSupplInAttrPSet* CG_inAttrPset, CG_NazeSORNUnitDataCollectorSupplOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==2);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
}

NazeSORNUnitDataCollectorSuppl::NazeSORNUnitDataCollectorSuppl() 
   : CG_NazeSORNUnitDataCollectorSuppl()
{
}

NazeSORNUnitDataCollectorSuppl::~NazeSORNUnitDataCollectorSuppl() 
{
}

void NazeSORNUnitDataCollectorSuppl::duplicate(std::unique_ptr<NazeSORNUnitDataCollectorSuppl>& dup) const
{
   dup.reset(new NazeSORNUnitDataCollectorSuppl(*this));
}

void NazeSORNUnitDataCollectorSuppl::duplicate(std::unique_ptr<Variable>& dup) const
{
   dup.reset(new NazeSORNUnitDataCollectorSuppl(*this));
}

void NazeSORNUnitDataCollectorSuppl::duplicate(std::unique_ptr<CG_NazeSORNUnitDataCollectorSuppl>& dup) const
{
   dup.reset(new NazeSORNUnitDataCollectorSuppl(*this));
}

