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
#include "CaCurrentDisplay.h"
#include "Simulation.h"
#include "CG_CaCurrentDisplay.h"
#include "MaxComputeOrder.h"
#include <memory>

#define decimal_places 3
#define fieldDelimiter "\t"

void CaCurrentDisplay::initialize(RNG& rng)
{
  if (ICa_channel.size() > 0 || ICa_synapse.size() > 0)
  {//Ca2+ current via Ca2+ channel; or Ca2+-permeable receptor at synapse
    assert(deltaT);
    std::ostringstream os;
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
    (*outFile) << "#Time" << fieldDelimiter << "CaCurrent :";
    if (ICa_channel.size() > 0)
    {//via Ca2+ channels
      if (indices.size() == 0)
      {
        for (unsigned int i = 0; i < ICa_channel.size(); ++i)
        {
          assert(ICa_channel[i]->size() > 0);
          for (unsigned int j = 0; j < ICa_channel[i]->size(); ++j)
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
        for (unsigned int i = 0; i < ICa_channel.size(); ++i)
        {
          for (it = indices.begin(); it != end; ++it)
          {
            if ((unsigned)*it < ICa_channel[i]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &channelBranchData[*it]->key)) << "," << *it
                         << "] ";
          }
        }
      }
    }
    if (ICa_synapse.size() > 0)
    {//via Ca2+-permeable receptor at synapse
      if (connexonBranchData.size() > 0)
      {
        for (unsigned int i = 0; i < ICa_synapse.size(); ++i)
        {
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &connexonBranchData[i]->key)) << ","
                     << *(connexonIndices[i]) << "] ";
        }
      }
      else
      {
        for (unsigned int i = 0; i < ICa_synapse.size(); ++i)
        {
          if (synapseIndices[i].size() != 2)
          {
            std::cout << "WRONG: synapseIndice.size() is " << synapseIndices[i].size() 
              << std::endl;
          }
          //NOTE: As the connectionn preCpt-SynapticCleft-postCpt
          // is established for every receptor on that SynapticCleft
          // then SynapticCleft->branchDataPrePost->size() is a multiple of 2
          //  times number of receptor-type on that cleft
          //  So we should not check this
          //assert(synapseBranchData[i]->size() == 2);
          //  But this is the right one
          assert(synapseBranchData[i]->size() % 2 == 0);

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

void CaCurrentDisplay::finalize(RNG& rng)
{
  if (outFile) outFile->close();
}

void CaCurrentDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (ICa_channel.size() > 0 || ICa_synapse.size() > 0)
  {
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (ICa_channel.size() > 0)
    {
      if (indices.size() == 0)
      {
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = ICa_channel
                                                                   .begin(),
                                                         end1 =
                                                             ICa_channel.end();
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
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = ICa_channel
                                                                   .begin(),
                                                         end1 =
                                                             ICa_channel.end();
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
    if (ICa_synapse.size() > 0)
    {
      ShallowArray<dyn_var_t*>::iterator it = ICa_synapse.begin(),
                                         end = ICa_synapse.end();
      for (; it != end; ++it)
      {
        (*outFile) << std::fixed << fieldDelimiter << (**it);
      }
    }
    (*outFile) << "\n";
  }
}

void CaCurrentDisplay::setUpPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CaCurrentDisplayInAttrPSet* CG_inAttrPset,
    CG_CaCurrentDisplayOutAttrPSet* CG_outAttrPset)
{
  if (CG_inAttrPset->identifier == "CHANNEL")
    ICa_channel.push_back(ICa_channelConnect);
  else if (CG_inAttrPset->identifier == "SYNAPSE")
  {
    synapseBranchData.push_back(synapseBranchDataConnect);
    synapseIndices.push_back(synapseIndicesConnect);
  }
}

CaCurrentDisplay::CaCurrentDisplay() : CG_CaCurrentDisplay(), outFile(0) {}

CaCurrentDisplay::~CaCurrentDisplay() { delete outFile; }

void CaCurrentDisplay::duplicate(std::auto_ptr<CaCurrentDisplay>& dup) const
{
  dup.reset(new CaCurrentDisplay(*this));
}

void CaCurrentDisplay::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new CaCurrentDisplay(*this));
}

void CaCurrentDisplay::duplicate(std::auto_ptr<CG_CaCurrentDisplay>& dup) const
{
  dup.reset(new CaCurrentDisplay(*this));
}
