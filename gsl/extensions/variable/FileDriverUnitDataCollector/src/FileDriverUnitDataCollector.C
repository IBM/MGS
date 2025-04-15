// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "FileDriverUnitDataCollector.h"
#include "CG_FileDriverUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void FileDriverUnitDataCollector::initialize(RNG& rng) 
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
  assert(slices.size()==outputs.size());
  int sz=outputs.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]][slices[j]]=outputs[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
    if (mxslice<slices[j]) mxslice=slices[j];
  }
  outputs.clear();
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
        outputs.push_back(miter3->second);
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
  
  std::ostringstream os_file;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;  

  if (op_saveOutputs)
    {
      os_file<<directory<<"Drivers"<<fileExt;
      output_file=new std::ofstream(os_file.str().c_str(),
                                    std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      output_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      output_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      output_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void FileDriverUnitDataCollector::finalize(RNG& rng) 
{
  if (op_saveOutputs)
    {
      output_file->close();
      delete output_file;
    }  
}

void FileDriverUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  ShallowArray<double*>::iterator iter, end;
  float temp = 0.;
  if (op_saveOutputs)
    {
      iter=outputs.begin();
      end=outputs.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          output_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }  
}

void FileDriverUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FileDriverUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_FileDriverUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

FileDriverUnitDataCollector::FileDriverUnitDataCollector() 
  : CG_FileDriverUnitDataCollector()
{
}

FileDriverUnitDataCollector::~FileDriverUnitDataCollector() 
{
}

void FileDriverUnitDataCollector::duplicate(std::unique_ptr<FileDriverUnitDataCollector>& dup) const
{
  dup.reset(new FileDriverUnitDataCollector(*this));
}

void FileDriverUnitDataCollector::duplicate(std::unique_ptr<Variable>& dup) const
{
  dup.reset(new FileDriverUnitDataCollector(*this));
}

void FileDriverUnitDataCollector::duplicate(std::unique_ptr<CG_FileDriverUnitDataCollector>& dup) const
{
  dup.reset(new FileDriverUnitDataCollector(*this));
}

