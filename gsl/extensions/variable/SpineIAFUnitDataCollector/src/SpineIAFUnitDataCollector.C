// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "SpineIAFUnitDataCollector.h"
#include "CG_SpineIAFUnitDataCollector.h"
#include "NodeDescriptor.h"
#include "Node.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

void SpineIAFUnitDataCollector::initialize(RNG& rng)
{
  // Sort pointers by indices, row major
  std::map<unsigned,
	   std::map<unsigned,
                    std::map<unsigned,
                             std::pair<
                               std::pair<double*, double*>, // second.first.first second.first.second
                               std::pair<
                                 // second.second.first
                                 std::pair<double*, // second.second.first.first
                                           double*> // second.second.first.second
                                 ,
                                 // second.second.second
                                 std::pair<double*, // second.second.second.first
                                           double*> // second.second.second.second
                                 >
                               >
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==AMPAcurrent.size());
  assert(slices.size()==mGluR5current.size());
  assert(slices.size()==NMDARopen.size());
  assert(slices.size()==NMDARCacurrent.size());
  assert(slices.size()==Ca.size());
  assert(slices.size()==eCB.size());
  int sz=AMPAcurrent.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(
                                                         std::make_pair(AMPAcurrent[j], mGluR5current[j]),
                                                         std::make_pair(
                                                                        std::make_pair(
                                                                                       NMDARopen[j]
                                                                                       ,
                                                                                       NMDARCacurrent[j]
                                                                                       )
                                                                        ,
                                                                        std::make_pair(
                                                                                       Ca[j]
                                                                                       ,
                                                                                       eCB[j]
                                                                                       )
                                                                        )
                                                         );
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  AMPAcurrent.clear();
  mGluR5current.clear();
  NMDARopen.clear();
  NMDARCacurrent.clear();
  Ca.clear();
  eCB.clear();
  std::map<unsigned,
	   std::map<unsigned,
                    std::map<unsigned,
                             std::pair<
                               std::pair<double*, double*>,
                               std::pair<
                                 std::pair<double*, double*>,
                                 std::pair<double*, double*>
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
                            std::pair<double*, double*>,
                            std::pair<double*, double*>
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
                       std::pair<double*, double*>,
                       std::pair<double*, double*>
                       >
                     >
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            {
              AMPAcurrent.push_back(miter3->second.first.first);
              mGluR5current.push_back(miter3->second.first.second);
              NMDARopen.push_back(miter3->second.second.first.first);
              NMDARCacurrent.push_back(miter3->second.second.first.second);
              Ca.push_back(miter3->second.second.second.first);
              eCB.push_back(miter3->second.second.second.second);
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

  std::ostringstream os_AMPA, os_mGluR5, os_mGluR5modulation, os_NMDARopen, os_NMDARCacurrent,
    os_Ca, os_eCBproduction, os_eCB;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;

  if (op_saveAMPA)
    {
      os_AMPA<<directory<<filePrep<<"SpineAMPA"<<fileApp<<fileExt;
      AMPA_file=new std::ofstream(os_AMPA.str().c_str(),
                                  std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      AMPA_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      AMPA_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      AMPA_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_savemGluR5)
    {
      // Save the mGluR5 modulation function first
      os_mGluR5modulation<<directory<<filePrep<<"SpinemGluR5modulation"
                         <<fileApp<<fileExt;
      mGluR5modulation_file=new std::ofstream(os_mGluR5modulation.str().c_str(),
                                              std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      double mGluR5 = 0.0;
      float CaModulation = 0.0;
      for (int i=0; i <= 2000; i++)
        {
          CaModulation = (float) mGluR5modulation(mGluR5);
          mGluR5modulation_file->write(reinterpret_cast<char *>(&CaModulation), sizeof(CaModulation));
          mGluR5 += 1.0 / 1000.0;
        }
      mGluR5modulation_file->close();
      delete mGluR5modulation_file;

      // Now the actual mGluR5 file
      os_mGluR5<<directory<<filePrep<<"SpinemGluR5"<<fileApp<<fileExt;
      mGluR5_file=new std::ofstream(os_mGluR5.str().c_str(),
                                    std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      mGluR5_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      mGluR5_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      mGluR5_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveNMDARopen)
    {
      os_NMDARopen<<directory<<filePrep<<"SpineNMDARopen"<<fileApp<<fileExt;
      NMDARopen_file=new std::ofstream(os_NMDARopen.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      NMDARopen_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      NMDARopen_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      NMDARopen_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveNMDARCacurrent)
    {
      os_NMDARCacurrent<<directory<<filePrep<<"SpineNMDARCacurrent"<<fileApp<<fileExt;
      NMDARCacurrent_file=new std::ofstream(os_NMDARCacurrent.str().c_str(),
                                            std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      NMDARCacurrent_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      NMDARCacurrent_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      NMDARCacurrent_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveCa)
    {
      os_Ca<<directory<<filePrep<<"SpineCa"<<fileApp<<fileExt;
      Ca_file=new std::ofstream(os_Ca.str().c_str(),
                                std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      Ca_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      Ca_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      Ca_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveeCB)
    {
      // Save the eCB production function first
      os_eCBproduction<<directory<<filePrep<<"SpineeCBproduction"<<fileApp<<fileExt;
      eCBproduction_file=new std::ofstream(os_eCBproduction.str().c_str(),
                                           std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      double Ca = 0.0;
      float ecb = 0.0;
      for (int i=0; i <= 2000; i++)
        {
          ecb = (float) eCBproduction(Ca);
          eCBproduction_file->write(reinterpret_cast<char *>(&ecb), sizeof(ecb));
          Ca += 1.0 / 1000.0;
        }
      eCBproduction_file->close();
      delete eCBproduction_file;

      // Now the actual eCB file
      os_eCB<<directory<<filePrep<<"SpineeCB"<<fileApp<<fileExt;
      eCB_file=new std::ofstream(os_eCB.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      eCB_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      eCB_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      eCB_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }
}

void SpineIAFUnitDataCollector::finalize(RNG& rng)
{
  // Close the output files...
  if (op_saveAMPA)
    {
      AMPA_file->close();
      delete AMPA_file;
    }
  if (op_savemGluR5)
    {
      mGluR5_file->close();
      delete mGluR5_file;
    }
  if (op_saveNMDARopen)
    {
      NMDARopen_file->close();
      delete NMDARopen_file;
    }
  if (op_saveNMDARCacurrent)
    {
      NMDARCacurrent_file->close();
      delete NMDARCacurrent_file;
    }
  if (op_saveCa)
    {
      Ca_file->close();
      delete Ca_file;
    }
  if (op_saveeCB)
    {
      eCB_file->close();
      delete eCB_file;
    }
}

void SpineIAFUnitDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (op_saveAMPA)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=AMPAcurrent.begin();
      end=AMPAcurrent.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          AMPA_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_savemGluR5)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=mGluR5current.begin();
      end=mGluR5current.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          mGluR5_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  
  if (op_saveNMDARopen)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=NMDARopen.begin();
      end=NMDARopen.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          NMDARopen_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
  
  if (op_saveNMDARCacurrent)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=NMDARCacurrent.begin();
      end=NMDARCacurrent.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          NMDARCacurrent_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_saveCa)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=Ca.begin();
      end=Ca.end();
      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          Ca_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }

  if (op_saveeCB)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=eCB.begin();
      end=eCB.end();

      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          eCB_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
}

void SpineIAFUnitDataCollector::getNodeIndices(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset)
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

SpineIAFUnitDataCollector::SpineIAFUnitDataCollector()
  : CG_SpineIAFUnitDataCollector()
{
}

SpineIAFUnitDataCollector::~SpineIAFUnitDataCollector()
{
}

void SpineIAFUnitDataCollector::duplicate(std::unique_ptr<SpineIAFUnitDataCollector>&& dup) const
{
  dup.reset(new SpineIAFUnitDataCollector(*this));
}

void SpineIAFUnitDataCollector::duplicate(std::unique_ptr<Variable>&& dup) const
{
  dup.reset(new SpineIAFUnitDataCollector(*this));
}

void SpineIAFUnitDataCollector::duplicate(std::unique_ptr<CG_SpineIAFUnitDataCollector>&& dup) const
{
  dup.reset(new SpineIAFUnitDataCollector(*this));
}


double SpineIAFUnitDataCollector::eCBsigmoid(double Ca)
{
  return 1.0 / ( 1.0 + exp(-eCBprodC * (Ca - eCBprodD)) );
}

double SpineIAFUnitDataCollector::eCBproduction(double Ca)
{
  // Computes the sigmoidal production of cannabinoids depending on Ca2+
  // NOTE: this is mirrored in SpineIAFUnitDataCollector. If changed here, change there too.
  double eCB = 0.0;
  // 1. the general sigmoid
  eCB = eCBsigmoid(Ca);
  // 2. make zero eCB at zero Ca2+
  eCB -= eCBsigmoid(0.0);
  // 3. Make one eCB at >= one Ca2+
  eCB *= 1.0 / (eCBsigmoid(1.0) - eCBsigmoid(0.0));
  if (eCB > 1.0)
    eCB = 1.0;

  return eCB;
}

double SpineIAFUnitDataCollector::mGluR5modulation(double mGluR5)
{
  return eCBproduction(mGluR5); // just use the same modified sigmoid
}

