// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "ConcentrationDisplay.h"
#include "CG_ConcentrationDisplay.h"
#include "Simulation.h"
#include "MaxComputeOrder.h"
#include <memory>

#define decimal_places 3
#define fieldDelimiter "\t"

void ConcentrationDisplay::initialize(RNG& rng)
{
  if (Conc_cpt.size() > 0 || Conc_synapse.size() > 0)
  {//Ca2+ current via Ca2+ channel; or Ca2+-permeable receptor at synapse
    assert(deltaT);
    std::ostringstream os;
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
    (*outFile) << "#Time" << fieldDelimiter << "Concentration :";
    if (Conc_cpt.size() > 0)
    {//via Ca2+ channels
      if (indices.size() == 0)
      {
        for (unsigned int i = 0; i < Conc_cpt.size(); ++i)
        {
          assert(Conc_cpt[i]->size() > 0);
          for (unsigned int j = 0; j < Conc_cpt[i]->size(); ++j)
          {
            (*outFile) << std::fixed << " ["
                       << *(reinterpret_cast<unsigned long long*>(
                              &channelBranchData[i]->key)) << "," << j << "] ";
          }
        }
      }
      else
      {
        ShallowArray<int>::iterator it, end = indices.end();
        int idx = 0;
        for (unsigned int i = 0; i < Conc_cpt.size(); ++i)
        {
          for (it = indices.begin(); it != end; ++it)
          {
            if ((unsigned)*it < Conc_cpt[i]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &channelBranchData[*it]->key)) << "," << *it
                         << "] ";
          }
        }
      }
    }
    if (Conc_synapse.size() > 0)
    {//via Ca2+-permeable receptor at synapse
      if (connexonBranchData.size() > 0)
      {
        for (unsigned int i = 0; i < Conc_synapse.size(); ++i)
        {
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &connexonBranchData[i]->key)) << ","
                     << *(connexonIndices[i]) << "] ";
        }
      }
      else
      {
        for (unsigned int i = 0; i < Conc_synapse.size(); ++i)
        {
          assert(synapseBranchData[i]->size() == 2);
					if (synapseIndices[i].size() != 2)
					{
						std::cout << "WRONG: synapseIndice.size() is " << synapseIndices[i].size() 
							<< std::endl;
					}

          assert(synapseIndices[i].size() == 2);
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &((*synapseBranchData[i])[0]->key))) << ","
                     << *(synapseIndices[i][0]) << "|"
                     << *(reinterpret_cast<unsigned long long*>(
                            &((*synapseBranchData[i])[1]->key))) << ","
                     << *(synapseIndices[i][1]) << "] ";
        }
      }
    }
    (*outFile) << "\n";
  }
}

void ConcentrationDisplay::finalize(RNG& rng)
{
  if (outFile) outFile->close();
}

void ConcentrationDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (Conc_cpt.size() > 0 || Conc_synapse.size() > 0)
  {
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (Conc_cpt.size() > 0)
    {
      if (indices.size() == 0)
      {
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = Conc_cpt
                                                                   .begin(),
                                                         end1 =
                                                             Conc_cpt.end();
        for (; it1 != end1; ++it1)
        {
          ShallowArray<dyn_var_t>::iterator it2 = (*it1)->begin(),
                                            end2 = (*it1)->end();
          for (; it2 != end2; ++it2) (*outFile) << std::fixed << fieldDelimiter << (*it2);
        }
      }
      else
      {
        ShallowArray<int>::iterator it2, end2 = indices.end();
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = Conc_cpt
                                                                   .begin(),
                                                         end1 =
                                                             Conc_cpt.end();
        for (; it1 != end1; ++it1)
        {
          for (it2 = indices.begin(); it2 != end2; ++it2)
          {
            if ((unsigned)*it2 < (*it1)->size())
              (*outFile) << std::fixed << fieldDelimiter << (**it1)[*it2];
          }
        }
      }
    }
    if (Conc_synapse.size() > 0)
    {
      ShallowArray<dyn_var_t*>::iterator it = Conc_synapse.begin(),
                                         end = Conc_synapse.end();
      for (; it != end; ++it)
      {
        (*outFile) << std::fixed << fieldDelimiter << (**it);
      }
    }
    (*outFile) << "\n";
  }
}

void ConcentrationDisplay::setUpPointers(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_ConcentrationDisplayInAttrPSet* CG_inAttrPset,
    CG_ConcentrationDisplayOutAttrPSet* CG_outAttrPset)
{
  if (CG_inAttrPset->identifier == "CHANNEL")
    Conc_cpt.push_back(Conc_cptConnect);
  else if (CG_inAttrPset->identifier == "SYNAPSE")
  {
    synapseBranchData.push_back(synapseBranchDataConnect);
    synapseIndices.push_back(synapseIndicesConnect);
  }
}

ConcentrationDisplay::ConcentrationDisplay() : CG_ConcentrationDisplay(), outFile(0) {}

ConcentrationDisplay::~ConcentrationDisplay() { delete outFile; }

void ConcentrationDisplay::duplicate(std::unique_ptr<ConcentrationDisplay>&& dup) const
{
  dup.reset(new ConcentrationDisplay(*this));
}

void ConcentrationDisplay::duplicate(std::unique_ptr<Variable>&& dup) const
{
  dup.reset(new ConcentrationDisplay(*this));
}

void ConcentrationDisplay::duplicate(std::unique_ptr<CG_ConcentrationDisplay>&& dup) const
{
  dup.reset(new ConcentrationDisplay(*this));
}
