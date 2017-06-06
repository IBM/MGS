// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CurrentDisplay.h"
#include "Simulation.h"
#include "CG_CurrentDisplay.h"
#include <memory>
#include "MaxComputeOrder.h"

void CurrentDisplay::initialize(RNG& rng)
{
  if (I_channel.size() > 0 || I_synapse.size() > 0)
  {
    assert(deltaT);
    std::ostringstream os;
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(3);
    (*outFile) << "#Time\tCurrent :";
    if (I_channel.size() > 0)
    {
      if (indices.size() == 0)
      {
        for (unsigned int i = 0; i < I_channel.size(); ++i)
        {
          assert(I_channel[i]->size() > 0);
          for (unsigned int j = 0; j < I_channel[i]->size(); ++j)
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
        for (unsigned int i = 0; i < I_channel.size(); ++i)
        {
          for (it = indices.begin(); it != end; ++it)
          {
            if ((unsigned)*it < I_channel[i]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &channelBranchData[*it]->key)) << "," << *it
                         << "] ";
          }
        }
      }
    }
    if (I_synapse.size() > 0)
    {
      if (connexonBranchData.size() > 0)
      {
        for (unsigned int i = 0; i < I_synapse.size(); ++i)
        {
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &connexonBranchData[i]->key)) << ","
                     << *(connexonIndices[i]) << "] ";
        }
      }
      else
      {
        for (unsigned int i = 0; i < I_synapse.size(); ++i)
        {
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

void CurrentDisplay::finalize(RNG& rng)
{
  if (outFile) outFile->close();
}

void CurrentDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (I_channel.size() > 0 || I_synapse.size() > 0)
  {
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (I_channel.size() > 0)
    {
      if (indices.size() == 0)
      {
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 =
                                                             I_channel.begin(),
                                                         end1 = I_channel.end();
        for (; it1 != end1; ++it1)
        {
          ShallowArray<dyn_var_t>::iterator it2 = (*it1)->begin(),
                                            end2 = (*it1)->end();
          for (; it2 != end2; ++it2) (*outFile) << std::fixed << "\t" << (*it2);
        }
      }
      else
      {
        ShallowArray<int>::iterator it2, end2 = indices.end();
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 =
                                                             I_channel.begin(),
                                                         end1 = I_channel.end();
        for (; it1 != end1; ++it1)
        {
          for (it2 = indices.begin(); it2 != end2; ++it2)
          {
            if ((unsigned)*it2 < (*it1)->size())
              (*outFile) << std::fixed << "\t" << (**it1)[*it2];
          }
        }
      }
    }
    if (I_synapse.size() > 0)
    {
      ShallowArray<dyn_var_t*>::iterator it = I_synapse.begin(),
                                         end = I_synapse.end();
      for (; it != end; ++it)
      {
        (*outFile) << std::fixed << "\t" << (**it);
      }
    }
    (*outFile) << "\n";
  }
}

void CurrentDisplay::setUpPointers(const String& CG_direction,
                                   const String& CG_component,
                                   NodeDescriptor* CG_node, Edge* CG_edge,
                                   VariableDescriptor* CG_variable,
                                   Constant* CG_constant,
                                   CG_CurrentDisplayInAttrPSet* CG_inAttrPset,
                                   CG_CurrentDisplayOutAttrPSet* CG_outAttrPset)
{
  if (CG_inAttrPset->identifier == "CHANNEL")
    I_channel.push_back(I_channelConnect);
  else if (CG_inAttrPset->identifier == "SYNAPSE")
  {
    synapseBranchData.push_back(synapseBranchDataConnect);
    synapseIndices.push_back(synapseIndicesConnect);
  }
}

CurrentDisplay::CurrentDisplay() : CG_CurrentDisplay(), outFile(0) {}

CurrentDisplay::~CurrentDisplay() { delete outFile; }

void CurrentDisplay::duplicate(std::auto_ptr<CurrentDisplay>& dup) const
{
  dup.reset(new CurrentDisplay(*this));
}

void CurrentDisplay::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new CurrentDisplay(*this));
}

void CurrentDisplay::duplicate(std::auto_ptr<CG_CurrentDisplay>& dup) const
{
  dup.reset(new CurrentDisplay(*this));
}
