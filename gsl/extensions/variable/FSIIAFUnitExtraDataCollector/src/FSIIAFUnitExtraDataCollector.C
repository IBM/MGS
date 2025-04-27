// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "FSIIAFUnitExtraDataCollector.h"
#include "CG_FSIIAFUnitExtraDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void FSIIAFUnitExtraDataCollector::initialize(RNG& rng) 
{
  // Sort pointers by indices, row major
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, 
                             std::pair< 
                               std::pair<float*, double*>, // second.first.first, second.first.second
                               std::pair< // second.second
                                 std::pair<
                                   double*, double*> // second.second.first.first, second.second.first.second
                                 ,
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
  assert(slices.size()==totalDriver.size());
  assert(slices.size()==totalIPSC.size());
  assert(slices.size()==totalGJ.size());
  int sz=voltages.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(
                                                       std::make_pair(voltages[j], thresholds[j]),
                                                       std::make_pair(
                                                                      std::make_pair(totalDriver[j], totalIPSC[j]),
                                                                      totalGJ[j]
                                                                      )
                                                       );
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
    if (mxslice<slices[j]) mxslice=slices[j];
  }
  voltages.clear();
  thresholds.clear();
  totalDriver.clear();
  totalIPSC.clear();
  totalGJ.clear();
  std::map<unsigned, 
	   std::map<unsigned, 
                    std::map<unsigned, 
                             std::pair< 
                               std::pair<float*, double*>,
                               std::pair<
                                 std::pair<double*, double*>,
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
                        std::pair<float*, double*>,
                        std::pair<
                          std::pair<double*, double*>,
                          double*
                          >
                        >
                      >
             >::iterator miter2, mend2=miter1->second.end();
    for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2) {
      std::map<unsigned,      
               std::pair< 
                 std::pair<float*, double*>,
                 std::pair<
                   std::pair<double*, double*>,
                   double*
                   >
                 >
               >::iterator miter3, mend3=miter2->second.end();
      for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3) {
        voltages.push_back(miter3->second.first.first);
        thresholds.push_back(miter3->second.first.second);
        totalDriver.push_back(miter3->second.second.first.first);
        totalIPSC.push_back(miter3->second.second.first.second);
        totalGJ.push_back(miter3->second.second.second);
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
  
  std::ostringstream os_voltages, os_thresholds, os_totalDriver, os_totalIPSC, os_totalGJ;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;  

  if (op_saveVoltages)
    {  
      os_voltages<<directory<<"Voltages"<<fileExt;
      voltages_file=new std::ofstream(os_voltages.str().c_str(),
                                      std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      voltages_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      voltages_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      voltages_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveThresholds)
    {  
      os_thresholds<<directory<<"Thresholds"<<fileExt;
      thresholds_file=new std::ofstream(os_thresholds.str().c_str(),
                                        std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      thresholds_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      thresholds_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      thresholds_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveTotalDriver)
    {  
      os_totalDriver<<directory<<"TotalDriver"<<fileExt;
      totalDriver_file=new std::ofstream(os_totalDriver.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      totalDriver_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      totalDriver_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      totalDriver_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveTotalIPSC)
    {  
      os_totalIPSC<<directory<<"TotalIPSC"<<fileExt;
      totalIPSC_file=new std::ofstream(os_totalIPSC.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      totalIPSC_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      totalIPSC_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      totalIPSC_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }  

  if (op_saveTotalGJ)
    {  
      os_totalGJ<<directory<<"TotalGJ"<<fileExt;
      totalGJ_file=new std::ofstream(os_totalGJ.str().c_str(),
                                     std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      totalGJ_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      totalGJ_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      totalGJ_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }  
}

void FSIIAFUnitExtraDataCollector::finalize(RNG& rng) 
{  
  if (op_saveVoltages)
    {
      voltages_file->close();
      delete voltages_file;
    }
  
  if (op_saveThresholds)
    {
      thresholds_file->close();
      delete thresholds_file;
    }
  
  if (op_saveTotalDriver)
    {
      totalDriver_file->close();
      delete totalDriver_file;
    }

  if (op_saveTotalIPSC)
    {
      totalIPSC_file->close();
      delete totalIPSC_file;
    }  
  
  if (op_saveTotalGJ)
    {
      totalGJ_file->close();
      delete totalGJ_file;
    }
}

void FSIIAFUnitExtraDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{

  if (op_saveVoltages)
    {
      ShallowArray<float*>::iterator iter3=voltages.begin(), end3=voltages.end();
      for (int n=0; iter3!=end3; ++iter3)
        voltages_file->write(reinterpret_cast<char *>(*iter3), sizeof(float));
    }
  
  ShallowArray<double*>::iterator iter, end;
  float temp = 0.;
  if (op_saveThresholds)
    {
      iter=thresholds.begin();
      end=thresholds.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          thresholds_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  
  if (op_saveTotalDriver)
    {
      iter=totalDriver.begin();
      end=totalDriver.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          totalDriver_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_saveTotalIPSC)
    {
      iter=totalIPSC.begin();
      end=totalIPSC.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          totalIPSC_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }  
  
  if (op_saveTotalGJ)
    {
      iter=totalGJ.begin();
      end=totalGJ.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          totalGJ_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  
}

void FSIIAFUnitExtraDataCollector::getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_FSIIAFUnitExtraDataCollectorInAttrPSet* CG_inAttrPset, CG_FSIIAFUnitExtraDataCollectorOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

FSIIAFUnitExtraDataCollector::FSIIAFUnitExtraDataCollector() 
  : CG_FSIIAFUnitExtraDataCollector()
{
}

FSIIAFUnitExtraDataCollector::~FSIIAFUnitExtraDataCollector() 
{
}

void FSIIAFUnitExtraDataCollector::duplicate(std::unique_ptr<FSIIAFUnitExtraDataCollector>&& dup) const
{
  dup.reset(new FSIIAFUnitExtraDataCollector(*this));
}

void FSIIAFUnitExtraDataCollector::duplicate(std::unique_ptr<Variable>duplicate(std::unique_ptr<Variable>& dup)duplicate(std::unique_ptr<Variable>& dup) dup) const
{
  dup.reset(new FSIIAFUnitExtraDataCollector(*this));
}

void FSIIAFUnitExtraDataCollector::duplicate(std::unique_ptr<CG_FSIIAFUnitExtraDataCollector>&& dup) const
{
  dup.reset(new FSIIAFUnitExtraDataCollector(*this));
}

