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
#include "ReversalPotentialDisplay.h"
#include "Simulation.h"
#include "CG_ReversalPotentialDisplay.h"
#include "MaxComputeOrder.h"
#include <memory>

void ReversalPotentialDisplay::initialize(RNG& rng)
{
  if (E_channel.size() > 0 || E_synapse.size() > 0)
  {
    assert(deltaT);
    std::ostringstream os;
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(3);
    (*outFile) << "#Time\tReversalPotential :";
    if (E_channel.size() > 0)
    {
      if (indices.size() == 0)
      {
        for (unsigned int i = 0; i < E_channel.size(); ++i)
        {
          if (E_channel[i]->size() == 0)
            std::cerr << "WARNING : No channel branch information available. "
                         "Consider initializing ReversalPotentialDisplay at a "
                         "later phase." << std::endl;
          for (unsigned int j = 0; j < E_channel[i]->size(); ++j)
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
        for (unsigned int i = 0; i < E_channel.size(); ++i)
        {
          for (it = indices.begin(); it != end; ++it)
          {
            if (E_channel[i]->size() == 0)
              std::cerr << "WARNING : No channel branch information available. "
                           "Consider initializing ReversalPotentialDisplay at "
                           "a later phase." << std::endl;
            if ((unsigned)*it < E_channel[i]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &channelBranchData[*it]->key)) << "," << *it
                         << "] ";
          }
        }
      }
    }
    if (E_synapse.size() > 0)
    {
      for (unsigned int i = 0; i < E_synapse.size(); ++i)
      {
        assert(synapseBranchData[i]->size() == 2);
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
    (*outFile) << "\n";
  }
}

void ReversalPotentialDisplay::finalize(RNG& rng)
{
  if (outFile) outFile->close();
}

void ReversalPotentialDisplay::dataCollection(Trigger* trigger,
                                              NDPairList* ndPairList)
{
  if (E_channel.size() > 0 || E_synapse.size() > 0)
  {
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (E_channel.size() > 0)
    {
      if (indices.size() == 0)
      {
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 =
                                                             E_channel.begin(),
                                                         end1 = E_channel.end();
        for (; it1 != end1; ++it1)
        {
          ShallowArray<dyn_var_t>::iterator it2 = (*it1)->begin(),
                                            end2 = (*it1)->end();
          assert((*it1)->size() > 0);
          for (; it2 != end2; ++it2) (*outFile) << std::fixed << "\t" << (*it2);
        }
      }
      else
      {
        ShallowArray<int>::iterator it2, end2 = indices.end();
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 =
                                                             E_channel.begin(),
                                                         end1 = E_channel.end();
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
    if (E_synapse.size() > 0)
    {
      ShallowArray<dyn_var_t*>::iterator it = E_synapse.begin(),
                                         end = E_synapse.end();
      for (; it != end; ++it)
      {
        (*outFile) << std::fixed << "\t" << (**it);
      }
    }
    (*outFile) << "\n";
  }
}

void ReversalPotentialDisplay::setUpPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_ReversalPotentialDisplayInAttrPSet* CG_inAttrPset,
    CG_ReversalPotentialDisplayOutAttrPSet* CG_outAttrPset)
{
  if (CG_inAttrPset->identifier == "SYNAPSE")
  {
    synapseBranchData.push_back(synapseBranchDataConnect);
    synapseIndices.push_back(synapseIndicesConnect);
  }
  else if (CG_inAttrPset->identifier == "CHANNEL")
  {
    E_channel.push_back(E_channelConnect);
  }
}

ReversalPotentialDisplay::ReversalPotentialDisplay()
    : CG_ReversalPotentialDisplay(), outFile(0)
{
}

ReversalPotentialDisplay::~ReversalPotentialDisplay() { delete outFile; }

void ReversalPotentialDisplay::duplicate(
    std::auto_ptr<ReversalPotentialDisplay>& dup) const
{
  dup.reset(new ReversalPotentialDisplay(*this));
}

void ReversalPotentialDisplay::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new ReversalPotentialDisplay(*this));
}

void ReversalPotentialDisplay::duplicate(
    std::auto_ptr<CG_ReversalPotentialDisplay>& dup) const
{
  dup.reset(new ReversalPotentialDisplay(*this));
}
