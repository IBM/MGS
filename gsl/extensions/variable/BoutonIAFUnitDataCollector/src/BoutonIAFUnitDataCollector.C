// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "BoutonIAFUnitDataCollector.h"
#include "CG_BoutonIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void BoutonIAFUnitDataCollector::initialize(RNG& rng)
{
  // Sort pointers by indices, row major
  std::map<unsigned,
	   std::map<unsigned,
                    std::map<unsigned,
                             std::pair< // second
                               std::pair< // second.first
                                 double*, // second.first.first
                                 double*> // second.first.second
                               ,
                               std::pair< // second.second
                                 std::pair< // second.second.first
                                   double*, // second.second.first.first
                                   double*> // second.second.first.second
                                 , 
                                 double*> // second.second.second
                               >
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==neurotransmitter.size());
  assert(slices.size()==availableNeurotransmitter.size());
  assert(slices.size()==CB1R.size());
  assert(slices.size()==CB1Runbound.size());
  assert(slices.size()==CB1Rcurrent.size());
  int sz=neurotransmitter.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(
                                                         std::make_pair(neurotransmitter[j]
                                                                        ,
                                                                        availableNeurotransmitter[j]
                                                                        )
                                                         ,
                                                         std::make_pair(
                                                                        std::make_pair(
                                                                                       CB1R[j]
                                                                                       ,
                                                                                       CB1Runbound[j]
                                                                                       )
                                                                        ,
                                                                        CB1Rcurrent[j]
                                                                        )
                                                         );
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  neurotransmitter.clear();
  availableNeurotransmitter.clear();
  CB1R.clear();
  CB1Runbound.clear();
  CB1Rcurrent.clear();
  std::map<unsigned,
	   std::map<unsigned,
                    std::map<unsigned,
                             std::pair<
                               std::pair<double*, double*>,
                               std::pair<
                                 std::pair<double*, double*>
                                 ,
                                 double*
                                 >
                               >
                             >
                    >
           >::iterator miter1, mend1=sorter.end();
  for (miter1=sorter.begin(); miter1!=mend1; ++miter1)
    {
      std::map<unsigned,
               std::map<unsigned,
                        std::pair<
                          std::pair<double*, double*>,
                          std::pair<
                            std::pair<double*, double*>
                            ,
                            double*
                            >
                          >
                        >
               >::iterator miter2, mend2=miter1->second.end();
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
        {
          std::map<unsigned,
                   std::pair<
                     std::pair<double*, double*>,
                     std::pair<
                       std::pair<double*, double*>
                       ,
                       double*
                       >
                     >
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            {
              neurotransmitter.push_back(miter3->second.first.first);
              availableNeurotransmitter.push_back(miter3->second.first.second);
              CB1R.push_back(miter3->second.second.first.first);
              CB1Runbound.push_back(miter3->second.second.first.second);
              CB1Rcurrent.push_back(miter3->second.second.second);
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

  std::ostringstream os_neurotransmitter, os_availableNeurotransmitter,
    os_CB1R, os_CB1Runbound, os_CB1Rcurrent;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;

  if (op_saveNeurotransmitter)
    {
      os_neurotransmitter<<directory<<filePrep<<"BoutonNeurotransmitter"<<fileApp<<fileExt;
      neurotransmitter_file=new std::ofstream(os_neurotransmitter.str().c_str(),
                                              std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      neurotransmitter_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      neurotransmitter_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      neurotransmitter_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveAvailableNeurotransmitter)
    {
      os_availableNeurotransmitter<<directory<<filePrep<<"BoutonAvailableNeurotransmitter"<<fileApp<<fileExt;
      availableNeurotransmitter_file=new std::ofstream(os_availableNeurotransmitter.str().c_str(),
                                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      availableNeurotransmitter_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      availableNeurotransmitter_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      availableNeurotransmitter_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveCB1R)
    {
      os_CB1R<<directory<<filePrep<<"BoutonCB1R"<<fileApp<<fileExt;
      CB1R_file=new std::ofstream(os_CB1R.str().c_str(),
                                  std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      CB1R_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      CB1R_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      CB1R_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveCB1Runbound)
    {
      os_CB1Runbound<<directory<<filePrep<<"BoutonCB1Runbound"<<fileApp<<fileExt;
      CB1Runbound_file=new std::ofstream(os_CB1Runbound.str().c_str(),
                                         std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      CB1Runbound_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      CB1Runbound_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      CB1Runbound_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

    

  if (op_saveCB1Rcurrent)
    {
      os_CB1Rcurrent<<directory<<filePrep<<"BoutonCB1Rcurrent"<<fileApp<<fileExt;
      CB1Rcurrent_file=new std::ofstream(os_CB1Rcurrent.str().c_str(),
                                         std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      CB1Rcurrent_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      CB1Rcurrent_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      CB1Rcurrent_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void BoutonIAFUnitDataCollector::finalize(RNG& rng)
{
  // Close the output files...
  if (op_saveNeurotransmitter)
    {
      neurotransmitter_file->close();
      delete neurotransmitter_file;
    }
  if (op_saveAvailableNeurotransmitter)
    {
      availableNeurotransmitter_file->close();
      delete availableNeurotransmitter_file;
    }
  if (op_saveCB1R)
    {
      CB1R_file->close();
      delete CB1R_file;
    }
  if (op_saveCB1Runbound)
    {
      CB1Runbound_file->close();
      delete CB1Runbound_file;
    }
  if (op_saveCB1Rcurrent)
    {
      CB1Rcurrent_file->close();
      delete CB1Rcurrent_file;
    }
}

void BoutonIAFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (op_saveNeurotransmitter)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=neurotransmitter.begin();
      end=neurotransmitter.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          neurotransmitter_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_saveAvailableNeurotransmitter)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=availableNeurotransmitter.begin();
      end=availableNeurotransmitter.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          availableNeurotransmitter_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_saveCB1R)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=CB1R.begin();
      end=CB1R.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          CB1R_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  
  if (op_saveCB1Runbound)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=CB1Runbound.begin();
      end=CB1Runbound.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          CB1Runbound_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  
  if (op_saveCB1Rcurrent)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=CB1Rcurrent.begin();
      end=CB1Rcurrent.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          CB1Rcurrent_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
}

void BoutonIAFUnitDataCollector::getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_BoutonIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_BoutonIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset)
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

BoutonIAFUnitDataCollector::BoutonIAFUnitDataCollector()
  : CG_BoutonIAFUnitDataCollector()
{
}

BoutonIAFUnitDataCollector::~BoutonIAFUnitDataCollector()
{
}

void BoutonIAFUnitDataCollector::duplicate(std::unique_ptr<BoutonIAFUnitDataCollector>&& dup) const
{
  dup.reset(new BoutonIAFUnitDataCollector(*this));
}

void BoutonIAFUnitDataCollector::duplicate(std::unique_ptr<Variable>duplicate(std::unique_ptr<Variable>& dup)duplicate(std::unique_ptr<Variable>& dup) dup) const
{
  dup.reset(new BoutonIAFUnitDataCollector(*this));
}

void BoutonIAFUnitDataCollector::duplicate(std::unique_ptr<CG_BoutonIAFUnitDataCollector>&& dup) const
{
  dup.reset(new BoutonIAFUnitDataCollector(*this));
}

