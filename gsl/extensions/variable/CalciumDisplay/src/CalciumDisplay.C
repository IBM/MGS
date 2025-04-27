// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "CalciumDisplay.h"
#include "Simulation.h"
#include "CG_CalciumDisplay.h"
#include "MaxComputeOrder.h"
#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

#include <memory>

#define decimal_places 5
#define fieldDelimiter "\t"

void CalciumDisplay::initialize(RNG& rng)
{
  if (Ca.size() > 0)
  {
    assert(deltaT);
    std::ostringstream os;
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
	// print out the header:
	// NOTE: The first field is always the time, and then next each field
	// represent the data from one specific recording compartment/channel
	// This compartment/channel can be uniquely identified using the
	// header information
	// The header each field has the format
	//[key-cpt,index-of-that-cpt](x,y,z,r,dist2soma) ...
	(*outFile) << "#Time" << fieldDelimiter << "Calcium :";
	if (indices.size() == 0)
    {
      for (unsigned int i = 0; i < dimensions.size(); ++i)
      {
        for (unsigned int j = 0; j < dimensions[i]->size(); ++j)
        {
					//TUAN TODO: potential BUG when key is not unsigned long long
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &branchData[i]->key)) << "," << j << "]("
                     << (*dimensions[i])[j]->x << "," << (*dimensions[i])[j]->y
                     << "," << (*dimensions[i])[j]->z << ","
                     << (*dimensions[i])[j]->r << ","
                     << (*dimensions[i])[j]->dist2soma << ") ";
        }
      }
    }
    else
    {
      ShallowArray<ShallowArray<int> >::iterator it1 = indices.begin(),
                                                 end1 = indices.end();
      for (int i = 0; it1 != end1; ++it1, ++i)
      {
        ShallowArray<int>::iterator it2 = it1->begin(), end2 = it1->end();
        for (; it2 != end2; ++it2)
        {
          assert(*it2 < dimensions[i]->size());
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &branchData[i]->key)) << "," << *it2 << "]("
                     << (*dimensions[i])[*it2]->x << ","
                     << (*dimensions[i])[*it2]->y << ","
                     << (*dimensions[i])[*it2]->z << ","
                     << (*dimensions[i])[*it2]->r << ","
                     << (*dimensions[i])[*it2]->dist2soma << ") ";
        }
      }
    }
    (*outFile) << "\n";
  }
}

void CalciumDisplay::finalize(RNG& rng)
{
  if (Ca.size() > 0) outFile->close();
}

void CalciumDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (Ca.size() > 0)
  {
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (indices.size() == 0)
    {
      ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = Ca.begin(),
                                                       end1 = Ca.end();
      for (; it1 != end1; ++it1)
      {
        ShallowArray<dyn_var_t>::iterator it2 = (*it1)->begin(),
                                          end2 = (*it1)->end();
        for (; it2 != end2; ++it2)
          (*outFile) << std::fixed << fieldDelimiter << (*it2);
      }
    }
    else
    {
      ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = Ca.begin(),
                                                       end1 = Ca.end();
      ShallowArray<ShallowArray<int> >::iterator it2 = indices.begin(),
                                                 end2 = indices.end();
      for (; it1 != end1; ++it1, ++it2)
      {
        assert(it2 != end2);
        ShallowArray<int>::iterator it3 = it2->begin(), end3 = it2->end();
        for (; it3 != end3; ++it3)
        {
          assert(*it3 < (*it1)->size());
          (*outFile) << std::fixed << fieldDelimiter << (**it1)[*it3];
        }
      }
    }
    (*outFile) << "\n";
  }
}

void CalciumDisplay::setUpPointers(const CustomString& CG_direction,
                                   const CustomString& CG_component,
                                   NodeDescriptor* CG_node, Edge* CG_edge,
                                   VariableDescriptor* CG_variable,
                                   Constant* CG_constant,
                                   CG_CalciumDisplayInAttrPSet* CG_inAttrPset,
                                   CG_CalciumDisplayOutAttrPSet* CG_outAttrPset)
{
  TissueSite& site = CG_inAttrPset->site;
  bool record = true;
  ShallowArray<int> ind;
  if (site.r != 0)
  {
    record = false;
    for (unsigned int i = 0; i < dimensions_connect->size(); ++i)
    {
      if ((site.r * site.r) >=
          DISTANCE_SQUARED(site, *((*dimensions_connect)[i])))
      {
        ind.push_back(i);
        record = true;
      }
    }
  }
  else if (indices.size() > 0)
  {
    for (unsigned int i = 0; i < dimensions_connect->size(); ++i) ind.push_back(i);
  }
  if (record)
  {
    if (ind.size() > 0)
    {
      if (indices.size() == 0)
      {
        for (unsigned int i = 0; i < Ca.size(); ++i)
        {
          ShallowArray<int> inds;
          for (unsigned int j = 0; j < Ca[i]->size(); ++j) inds.push_back(j);
          indices.push_back(inds);
        }
      }
      indices.push_back(ind);
    }
    dimensions.push_back(dimensions_connect);
    Ca.push_back(Ca_connect);
    branchData.push_back(branchData_connect);
  }
}

CalciumDisplay::CalciumDisplay() : CG_CalciumDisplay(), outFile(0) {}

CalciumDisplay::~CalciumDisplay() { delete outFile; }

void CalciumDisplay::duplicate(std::unique_ptr<CalciumDisplay>&& dup) const
{
  dup.reset(new CalciumDisplay(*this));
}

void CalciumDisplay::duplicate(std::unique_ptr<Variable>duplicate(std::unique_ptr<Variable>& dup)duplicate(std::unique_ptr<Variable>& dup) dup) const
{
  dup.reset(new CalciumDisplay(*this));
}

void CalciumDisplay::duplicate(std::unique_ptr<CG_CalciumDisplay>&& dup) const
{
  dup.reset(new CalciumDisplay(*this));
}
