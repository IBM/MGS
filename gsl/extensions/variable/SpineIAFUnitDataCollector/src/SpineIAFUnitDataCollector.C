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
                               std::pair<double*, double*>  // second.second.first second.second.second
                               >
                             >
                    >
           >
    sorter;
  assert(rows.size()==slices.size());
  assert(cols.size()==slices.size());
  assert(slices.size()==AMPAcurrent.size());
  assert(slices.size()==mGluR5current.size());
  assert(slices.size()==Ca.size());
  assert(slices.size()==ECB.size());
  int sz=AMPAcurrent.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j)
    {
      sorter[rows[j]][cols[j]][slices[j]]=std::make_pair(
                                                         std::make_pair(AMPAcurrent[j], mGluR5current[j]),
                                                         std::make_pair(Ca[j], ECB[j])
                                                         );
      if (mxrow<rows[j]) mxrow=rows[j];
      if (mxcol<cols[j]) mxcol=cols[j];
      if (mxslice<slices[j]) mxslice=slices[j];
    }
  AMPAcurrent.clear();
  mGluR5current.clear();
  Ca.clear();
  ECB.clear();
  std::map<unsigned,
	   std::map<unsigned,
                    std::map<unsigned,
                             std::pair<
                               std::pair<double*, double*>,
                               std::pair<double*, double*>
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
                          std::pair<double*, double*>
                          >
                        >
               >::iterator miter2, mend2=miter1->second.end();
      for (miter2=miter1->second.begin(); miter2!=mend2; ++miter2)
        {
          std::map<unsigned,
                   std::pair<
                     std::pair<double*, double*>,
                     std::pair<double*, double*>
                     >
                   >::iterator miter3, mend3=miter2->second.end();
          for (miter3=miter2->second.begin(); miter3!=mend3; ++miter3)
            {
              AMPAcurrent.push_back(miter3->second.first.first);
              mGluR5current.push_back(miter3->second.first.second);
              Ca.push_back(miter3->second.second.first);
              ECB.push_back(miter3->second.second.second);
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

  std::ostringstream os_AMPA, os_mGluR5, os_mGluR5modulation, os_Ca, os_ECBproduction, os_ECB;

  int Xdim = (int) mxslice+1;
  int Ydim = (int) mxcol+1;
  int Zdim = (int) mxrow+1;

  if (op_saveAMPA)
    {
      os_AMPA<<directory<<"AMPA"<<fileExt;
      AMPA_file=new std::ofstream(os_AMPA.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      AMPA_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      AMPA_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      AMPA_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_savemGluR5)
    {
      // Save the mGluR5 modulation function first
      os_mGluR5modulation<<directory<<"mGluR5modulation"<<fileExt;
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
      os_mGluR5<<directory<<"mGluR5"<<fileExt;
      mGluR5_file=new std::ofstream(os_mGluR5.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      mGluR5_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      mGluR5_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      mGluR5_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveCa)
    {
      os_Ca<<directory<<"Ca"<<fileExt;
      Ca_file=new std::ofstream(os_Ca.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      Ca_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      Ca_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      Ca_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
    }

  if (op_saveECB)
    {
      // Save the ECB production function first
      os_ECBproduction<<directory<<"ECBproduction"<<fileExt;
      ECBproduction_file=new std::ofstream(os_ECBproduction.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      double Ca = 0.0;
      float ecb = 0.0;
      for (int i=0; i <= 2000; i++)
        {
          ecb = (float) ECBproduction(Ca);
          ECBproduction_file->write(reinterpret_cast<char *>(&ecb), sizeof(ecb));
          Ca += 1.0 / 1000.0;
        }
      ECBproduction_file->close();
      delete ECBproduction_file;

      // Now the actual ECB file
      os_ECB<<directory<<"ECB"<<fileExt;
      ECB_file=new std::ofstream(os_ECB.str().c_str(),
                                       std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      ECB_file->write(reinterpret_cast<char *>(&Xdim), sizeof(Xdim));
      ECB_file->write(reinterpret_cast<char *>(&Ydim), sizeof(Ydim));
      ECB_file->write(reinterpret_cast<char *>(&Zdim), sizeof(Zdim));
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
  if (op_saveCa)
    {
      Ca_file->close();
      delete Ca_file;
    }
  if (op_saveECB)
    {
      ECB_file->close();
      delete ECB_file;
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

  if (op_saveECB)
    {
      ShallowArray<double*>::iterator iter, end;
      float temp = 0.;
      iter=ECB.begin();
      end=ECB.end();

      for (int n=0; iter!=end; ++iter)
        {
          temp = (float) **iter;
          ECB_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
}

void SpineIAFUnitDataCollector::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_SpineIAFUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_SpineIAFUnitDataCollectorOutAttrPSet* CG_outAttrPset)
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

void SpineIAFUnitDataCollector::duplicate(std::auto_ptr<SpineIAFUnitDataCollector>& dup) const
{
   dup.reset(new SpineIAFUnitDataCollector(*this));
}

void SpineIAFUnitDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new SpineIAFUnitDataCollector(*this));
}

void SpineIAFUnitDataCollector::duplicate(std::auto_ptr<CG_SpineIAFUnitDataCollector>& dup) const
{
   dup.reset(new SpineIAFUnitDataCollector(*this));
}


double SpineIAFUnitDataCollector::ECBsigmoid(double Ca)
{
  return 1.0 / ( 1.0 + exp(-ECBprodC * (Ca - ECBprodD)) );
}

double SpineIAFUnitDataCollector::ECBproduction(double Ca)
{
  // Computes the sigmoidal production of cannabinoids depending on Ca2+
  // NOTE: this is mirrored in SpineIAFUnitDataCollector. If changed here, change there too.
  double ECB = 0.0;
  // 1. the general sigmoid
  ECB = ECBsigmoid(Ca);
  // 2. make zero ECB at zero Ca2+
  ECB -= ECBsigmoid(0.0);
  // 3. Make one ECB at >= one Ca2+
  ECB *= 1.0 / (ECBsigmoid(1.0) - ECBsigmoid(0.0));
  if (ECB > 1.0)
    ECB = 1.0;

  return ECB;
}

double SpineIAFUnitDataCollector::mGluR5modulation(double mGluR5)
{
  return ECBproduction(mGluR5); // just use the same modified sigmoid
}

