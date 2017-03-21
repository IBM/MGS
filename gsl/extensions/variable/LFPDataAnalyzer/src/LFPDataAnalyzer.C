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
#include "LFPDataAnalyzer.h"
#include "CG_LFPDataAnalyzer.h"
#include "Node.h"
#include "NodeDescriptor.h"
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

double LFPDataAnalyzer::normal_pdf(double x, double mean, double sigma)
{
  double invSqrt2Pi = 0.3989422804014327;
  double tmp = (x - mean) / sigma;
  return (invSqrt2Pi / sigma) * exp(-0.5 * tmp * tmp);
}

void LFPDataAnalyzer::initialize(RNG& rng)
{
  // 1. Setup the collection of individual LFPs
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
  assert(slices.size()==LFPs_individual.size());
  int sz=LFPs_individual.size();
  int mxrow=0;
  int mxcol=0;
  for (int j=0; j<sz; ++j) {
    sorter[rows[j]][cols[j]][slices[j]]=LFPs_individual[j];
    if (mxrow<rows[j]) mxrow=rows[j];
    if (mxcol<cols[j]) mxcol=cols[j];
    if (mxslice<slices[j]) mxslice=slices[j];
  }
  LFPs_individual.clear();
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
        LFPs_individual.push_back(miter3->second);
      }
    }
  }


  
  // 2. Setup the electrode positioning
  Xdim = (int) mxslice+1;
  Ydim = (int) mxcol+1;
  Zdim = (int) mxrow+1;  
  
  elecCenterX.increaseSizeTo(numElecPerDimX);
  elecCenterY.increaseSizeTo(numElecPerDimY);
  elecCenterZ.increaseSizeTo(numElecPerDimZ);
  
  for (int i=0; i < numElecPerDimX; i++)
    elecCenterX[i] = ((double) Xdim / (double) (numElecPerDimX + 1)) * (double) (i + 1);
  for (int i=0; i < numElecPerDimY; i++)
    elecCenterY[i] = ((double) Ydim / (double) (numElecPerDimY + 1)) * (double) (i + 1);
  for (int i=0; i < numElecPerDimZ; i++)
    elecCenterZ[i] = ((double) Zdim / (double) (numElecPerDimZ + 1)) * (double) (i + 1);
  
  LFPs.increaseSizeTo(numElecPerDimX * numElecPerDimY * numElecPerDimZ);

  
    
  // 3. Create the output files...
  std::ostringstream sysCall;
  sysCall<<"mkdir -p "<<directory.c_str()<<";";
  try {
    int systemRet = system(sysCall.str().c_str());
    if (systemRet == -1)
      throw;
  } catch(...) {};
  
  std::ostringstream os_LFP;

  if (op_saveLFPs)
    {
      os_LFP<<directory<<"LFP"<<fileExt;
      LFP_file=new std::ofstream(os_LFP.str().c_str(),
                                 std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
      LFP_file->write(reinterpret_cast<char *>(&numElecPerDimX), sizeof(numElecPerDimX));
      LFP_file->write(reinterpret_cast<char *>(&numElecPerDimY), sizeof(numElecPerDimY));
      LFP_file->write(reinterpret_cast<char *>(&numElecPerDimZ), sizeof(numElecPerDimZ));
    }
}

void LFPDataAnalyzer::finalize(RNG& rng) 
{
  if (op_saveLFPs)
    {
      LFP_file->close();
      delete LFP_file;
    }  
}

void LFPDataAnalyzer::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  // TODO: can probably made more efficient. But this brute force loop over
  // everything approach is good enough for now.
  if (op_saveLFPs)
    {
      ShallowArray<int>::iterator iterRows, endRows=rows.end();
      ShallowArray<int>::iterator iterCols, endCols=cols.end();
      ShallowArray<int>::iterator iterSlices, endSlices=slices.end();
      ShallowArray<double>::iterator iterLFPs=LFPs.begin(), endLFPs=LFPs.end();
      ShallowArray<double*>::iterator iterLFPs_ind,
        endLFPs_ind=LFPs_individual.end();
      double dist;
      // For each electrode ...
      for (int z=0; z<numElecPerDimZ; z++)
        {
          for (int y=0; y<numElecPerDimY; y++)
            {
              for (int x=0; x<numElecPerDimX; x++)
                {
                  iterLFPs_ind = LFPs_individual.begin();
                  iterRows = rows.begin();
                  iterCols = cols.begin();
                  iterSlices = slices.begin();
                  *iterLFPs = 0.0;
                  // ... work out the contribution of the LFP
                  // with a 3D Gaussian weighting depending on distance
                  // to the elctrode.
                  for (int n=0; iterLFPs_ind != endLFPs_ind;
                       ++iterLFPs_ind, ++iterRows, ++iterCols, ++iterSlices)
                    {
                      dist = sqrt(
                                  pow(elecCenterX[x]-*iterRows,2)
                                  + pow(elecCenterY[y]-*iterCols,2)
                                  + pow(elecCenterZ[z]-*iterSlices,2)
                                  );
                      if (dist <= elecRadius)
                        *iterLFPs += normal_pdf(dist, 0.0, elecSigma) * **iterLFPs_ind;
                    }
                  ++iterLFPs;
                }
            }
        }
      // Save these LFPs to disk
      float temp;
      for(iterLFPs=LFPs.begin(); iterLFPs != endLFPs; ++iterLFPs)
        {
          temp = (float) *iterLFPs;
          LFP_file->write(reinterpret_cast<char *>(&temp), sizeof(temp));
        }
    }
}

void LFPDataAnalyzer::getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_LFPDataAnalyzerInAttrPSet* CG_inAttrPset, CG_LFPDataAnalyzerOutAttrPSet* CG_outAttrPset) 
{
  ShallowArray<unsigned,3,2> coords;
  CG_node->getNode()->getNodeCoords(coords);
  assert(coords.size()==3);
  rows.push_back(coords[0]);
  cols.push_back(coords[1]);
  slices.push_back(coords[2]);
}

LFPDataAnalyzer::LFPDataAnalyzer() 
  : CG_LFPDataAnalyzer()
{
}

LFPDataAnalyzer::~LFPDataAnalyzer() 
{
}

void LFPDataAnalyzer::duplicate(std::auto_ptr<LFPDataAnalyzer>& dup) const
{
  dup.reset(new LFPDataAnalyzer(*this));
}

void LFPDataAnalyzer::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new LFPDataAnalyzer(*this));
}

void LFPDataAnalyzer::duplicate(std::auto_ptr<CG_LFPDataAnalyzer>& dup) const
{
  dup.reset(new LFPDataAnalyzer(*this));
}

