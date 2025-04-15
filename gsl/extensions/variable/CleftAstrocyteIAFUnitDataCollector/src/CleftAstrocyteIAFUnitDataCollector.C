// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "CleftAstrocyteIAFUnitDataCollector.h"
#include "CG_CleftAstrocyteIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void CleftAstrocyteIAFUnitDataCollector::initialize(RNG& rng) 
{  
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair<double*, double*> // second.first second.second
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==neurotransmitter.size());
  assert(slices.size()==eCB.size());
  int sz=neurotransmitter.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(neurotransmitter[j], eCB[j]);
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  neurotransmitter.clear();
  eCB.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned,
                             std::pair<double*, double*>
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
    {
      std::map<unsigned, 
               std::map<unsigned,
                        std::pair<double*, double*>
                        >
               >::iterator miter2, mend2=miter1->second.end();    
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
        {
          std::map<unsigned,
                   std::pair<double*, double*>
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            {
              neurotransmitter.push_back(miter3->second.first);
              eCB.push_back(miter3->second.second);
            }
        }
    }

  // Create the output files...
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<directory.c_str()<<";";
  try
    {    
      int systemRet = system(sysCall.str().c_str());
      if (systemRet == -1)
        throw;
    }
  catch(...) { };
  
  std::ostringstream os_neurotransmitter, os_eCB;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;

  if (op_saveNeurotransmitter)
    {
      os_neurotransmitter<<directory<<filePrep<<"CleftAstrocyteNeurotransmitter"
                         <<fileApp<<fileExt;
      neurotransmitter_file=new std::ofstream(os_neurotransmitter.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      neurotransmitter_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      neurotransmitter_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      neurotransmitter_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveeCB)
    {
      os_eCB<<directory<<filePrep<<"CleftAstrocyteeCB"<<fileApp<<fileExt;
      eCB_file=new std::ofstream(os_eCB.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      eCB_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      eCB_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      eCB_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }      
}

void CleftAstrocyteIAFUnitDataCollector::finalize(RNG& rng) 
{
  // Close the output files...
  if (op_saveNeurotransmitter)
    {
      neurotransmitter_file->close();
      delete neurotransmitter_file;
    }  
  if (op_saveeCB)
    {
      eCB_file->close();
      delete eCB_file;
    }  
}

void CleftAstrocyteIAFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (op_saveNeurotransmitter)
    {
      ShallowArray<double*>::iterator iter=neurotransmitter.begin(), end=neurotransmitter.end();
      float temp = 0.;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = (float) **iter;
          neurotransmitter_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));            
        }
    }

  if (op_saveeCB)
    {
      ShallowArray<double*>::iterator iter=eCB.begin(), end=eCB.end();
      float temp = 0.;
      for (int n=0; iter!=end; ++iter, n++)
        {
          temp = (float) **iter;
          eCB_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));            
        }
    }  
}

void CleftAstrocyteIAFUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

CleftAstrocyteIAFUnitDataCollector::CleftAstrocyteIAFUnitDataCollector() 
  : CG_CleftAstrocyteIAFUnitDataCollector()
{
}

CleftAstrocyteIAFUnitDataCollector::~CleftAstrocyteIAFUnitDataCollector() 
{
}

void CleftAstrocyteIAFUnitDataCollector::duplicate(std::unique_ptr<CleftAstrocyteIAFUnitDataCollector>& dup) const
{
  dup.reset(new CleftAstrocyteIAFUnitDataCollector(*this));
}

void CleftAstrocyteIAFUnitDataCollector::duplicate(std::unique_ptr<Variable>& dup) const
{
  dup.reset(new CleftAstrocyteIAFUnitDataCollector(*this));
}

void CleftAstrocyteIAFUnitDataCollector::duplicate(std::unique_ptr<CG_CleftAstrocyteIAFUnitDataCollector>& dup) const
{
  dup.reset(new CleftAstrocyteIAFUnitDataCollector(*this));
}

