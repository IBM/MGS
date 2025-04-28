// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "VoltageDisplay.h"
#include "Simulation.h"
#include "CG_VoltageDisplay.h"
#include "MaxComputeOrder.h"
#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

#include <memory>

#define decimal_places 3
#define fieldDelimiter "\t"

void VoltageDisplay::initialize(RNG& rng)
{
  if (V.size() > 0)
  {
    assert(deltaT);
    std::ostringstream os;
		//file name: filename_<MPIrank>
		//e.g. somaVm.dat_0
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
	// print out the header:
	// NOTE: The first field is always the time, and then next each field
	// represent the data from one specific recording compartment/channel
	// This compartment/channel can be uniquely identified using the
	// header information
	// The header each field has the format
	//[key-cpt,index-of-that-cpt](x,y,z,r,dist2soma) ...
	outFile->precision(decimal_places);
    (*outFile) << "#Time" << fieldDelimiter << "Voltage :";
    if (indices.size() == 0)
    {
      for (unsigned int i = 0; i < dimensions.size(); ++i)
      {
        for (unsigned int j = 0; j < dimensions[i]->size(); ++j)
        {
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
					//[key_cpt, index-value-in-that-cpt](x,y,z,r,dist2soma)
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

void VoltageDisplay::finalize(RNG& rng)
{
  if (V.size() > 0) outFile->close();
}

// TODO: TUAN - put << std.endl; 
//  after a certain interval for data write out
void VoltageDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (V.size() > 0)
  {
    //float current_time = float(getSimulation().getIteration()) * *deltaT; // [msec]
    float current_time = ((double)getSimulation().getIteration()-1) * *deltaT; // [msec]
    current_time += (*deltaT)/2.0;  //second-order accuracy at time t+dt/2
    (*outFile) << current_time; 
    if (indices.size() == 0)
    {//perform I/O routine
      ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = V.begin(),
        end1 = V.end();
      for (; it1 != end1; ++it1)
      {
        ShallowArray<dyn_var_t>::iterator it2 = (*it1)->begin(),
          end2 = (*it1)->end();
        for (; it2 != end2; ++it2) (*outFile) << std::fixed << fieldDelimiter << (*it2);
      }
    }
    else
    {
      ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = V.begin(),
        end1 = V.end();
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
    //if (every second )
    // call flush
    // else
    (*outFile) << "\n";
  }
}

void VoltageDisplay::setUpPointers(const CustomString& CG_direction,
    const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge,
    VariableDescriptor* CG_variable,
    Constant* CG_constant,
    CG_VoltageDisplayInAttrPSet* CG_inAttrPset,
    CG_VoltageDisplayOutAttrPSet* CG_outAttrPset)
{
  TissueSite& site = CG_inAttrPset->site;
  bool record = true;
  ShallowArray<int> ind;
  if (site.r != 0)
  {//record based on distance criteria to a site
    record = false;
    for (unsigned int i = 0; i < dimensions_connect->size(); ++i)
    {//make sure it connect to the 'whatever-data' associated with the compartment within the 
      // spherical range
      if ((site.r * site.r) >=
          DISTANCE_SQUARED(site, *((*dimensions_connect)[i])))
      {
        ind.push_back(i);
        record = true;
      }
    }
  }
  else if (indices.size() > 0)
  {// record data from all compartments associated with the given ComputeBranch
    for (unsigned int i = 0; i < dimensions_connect->size(); ++i)
      ind.push_back(i);
  }

  //now check
  if (record)
  {
    if (ind.size() > 0)
    {
      if (indices.size() == 0)
      {// TUAN TODO: Not sure when this happens (need to ask James) 
        for (unsigned int i = 0; i < V.size(); ++i)
        {
          ShallowArray<int> inds;
          for (unsigned int j = 0; j < V[i]->size(); ++j) inds.push_back(j);
          indices.push_back(inds);
        }
      }
      indices.push_back(ind);
    }
    dimensions.push_back(dimensions_connect);
    V.push_back(V_connect);
    branchData.push_back(branchData_connect);
  }
}

VoltageDisplay::VoltageDisplay() : CG_VoltageDisplay(), outFile(0) {}

VoltageDisplay::~VoltageDisplay() { delete outFile; }

void VoltageDisplay::duplicate(std::unique_ptr<VoltageDisplay>&& dup) const
{
  dup.reset(new VoltageDisplay(*this));
}

void VoltageDisplay::duplicate(std::unique_ptr<Variable>&& dup) const
{
  dup.reset(new VoltageDisplay(*this));
}

void VoltageDisplay::duplicate(std::unique_ptr<CG_VoltageDisplay>&& dup) const
{
  dup.reset(new VoltageDisplay(*this));
}
