#include "Lens.h"
#include "AnyCurrentDisplay.h"
#include "CG_AnyCurrentDisplay.h"
#include "MaxComputeOrder.h"
#include <memory>

#define decimal_places 3
#define fieldDelimiter "\t"

void AnyCurrentDisplay::initialize(RNG& rng)
{
  if (I_channel.size() > 0 || I_synapse.size() > 0
#ifdef IDEA_ILEAK
      || V.size() > 0
#endif
      )
  {  // current via Ca2+/Na+/or K+ channel; or ion/Ca2+-permeable receptor at
     // synapse
    assert(deltaT);
    std::ostringstream os;
    std::ostringstream os2;
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
    (*outFile) << "#Time" << fieldDelimiter << "AnyCurrent :";
    os2 << "# time, ";
    if (I_channel.size() > 0)
    {  // via channels
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
              os2 << strChannelTypeOnCompt[i] << ", "; 
          }
        }
      }
      else
      {
        ShallowArray<int>::iterator it, end = indices.end();
        for (unsigned int i = 0; i < I_channel.size(); ++i)
        {
          for (it = indices.begin(); it != end; ++it)
          {
            if ((unsigned)*it < I_channel[i]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &channelBranchData[*it]->key)) << "," << *it
                         << "] ";
              os2 << strChannelTypeOnCompt[i] << ", "; 
          }
        }
      }
    }
    if (I_synapse.size() > 0)
    {  // via ion-permeable receptor at synapse
      if (connexonBranchData.size() > 0)
      {
        for (unsigned int i = 0; i < I_synapse.size(); ++i)
        {
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &connexonBranchData[i]->key)) << ","
                     << *(connexonIndices[i]) << "] ";
              os2 << strChannelTypeOnSynapse[i] << ", "; 
        }
      }
      else
      {
        for (unsigned int i = 0; i < I_synapse.size(); ++i)
        {
          if (synapseIndices[i].size() != 2)
          {
            std::cout << "WRONG: synapseIndice.size() is "
                      << synapseIndices[i].size() << std::endl;
          }
          assert(synapseBranchData[i]->size() == 2);

          assert(synapseIndices[i].size() == 2);
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &((*synapseBranchData[i])[0]->key))) << ","
                     << *(synapseIndices[i][0]) << "|"
                     << *(reinterpret_cast<unsigned long long*>(
                            &((*synapseBranchData[i])[1]->key))) << ","
                     << *(synapseIndices[i][1]) << "] ";
          os2 << strChannelTypeOnSynapse[i] << ", "; 
        }
      }
    }
#ifdef IDEA_ILEAK
    if (V.size() > 0)
    {
      if (indices.size() == 0)
      {
        for (unsigned int i = 0; i < V.size(); ++i)
        {
          assert(V[i]->size() > 0);
          for (unsigned int j = 0; j < V[i]->size(); ++j)
          {
            (*outFile) << std::fixed << " ["
                       << *(reinterpret_cast<unsigned long long*>(
                              &leakBranchData[i]->key)) << "," << j << "] ";
              os2 << "leak" << ", "; 
          }
        }
      }
      else
      {
        ShallowArray<int>::iterator it, end = indices.end();
        for (unsigned int i = 0; i < V.size(); ++i)
        {
          for (it = indices.begin(); it != end; ++it)
          {
            if ((unsigned)*it < V[i]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &leakBranchData[*it]->key)) << "," << *it
                         << "] ";
              os2 << "leak" << ", "; 
          }
        }
      }
      
    }
#endif
    (*outFile) << "\n";
    (*outFile) << os2.str() << "\n";
  }
}

void AnyCurrentDisplay::finalize(RNG& rng)
{
  if (outFile) outFile->close();
}

void AnyCurrentDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (I_channel.size() > 0 || I_synapse.size() > 0
#ifdef IDEA_ILEAK
      || V.size() > 0
#endif
      )
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
          for (; it2 != end2; ++it2)
            (*outFile) << std::fixed << fieldDelimiter << (*it2);
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
              (*outFile) << std::fixed << fieldDelimiter << (**it1)[*it2];
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
        (*outFile) << std::fixed << fieldDelimiter << (**it);
      }
    }
#ifdef IDEA_ILEAK
    if (V.size() > 0)
    {
      if (indices.size() == 0)
      {
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 =
                                                             V.begin(),
                                                         end1 = V.end();
        for (; it1 != end1; ++it1)
        {
          ShallowArray<dyn_var_t>::iterator it2 = (*it1)->begin(),
                                            end2 = (*it1)->end();
          int i=0;
          for (; it2 != end2; ++it2)
          {
            float Ileak = *(gLeak[i]) * ((*it2) - *(Eleak[i]));
            (*outFile) << std::fixed << fieldDelimiter << Ileak ;
            
          }
        }
      }
      else
      {
        ShallowArray<int>::iterator it2, end2 = indices.end();
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 =
                                                             V.begin(),
                                                         end1 = V.end();
        for (; it1 != end1; ++it1)
        {
          for (it2 = indices.begin(); it2 != end2; ++it2)
          {
            if ((unsigned)*it2 < (*it1)->size())
            {
              float Ileak = *(gLeak[*it2]) * ((**it1)[*it2] - *(Eleak[*it2]));
              (*outFile) << std::fixed << fieldDelimiter << Ileak;
            }
          }
        }
      }
    }
#endif
    (*outFile) << "\n";
  }
}

void AnyCurrentDisplay::setUpPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_AnyCurrentDisplayInAttrPSet* CG_inAttrPset,
    CG_AnyCurrentDisplayOutAttrPSet* CG_outAttrPset)
{
  String type(CG_inAttrPset->type);
  if (type == "")
    type = "unknown";
  if (CG_inAttrPset->identifier == "CHANNEL")
  {
    I_channel.push_back(I_channelConnect);
    strChannelTypeOnCompt.push_back(type);
  }
  else if (CG_inAttrPset->identifier == "SYNAPSE")
  {
    synapseBranchData.push_back(synapseBranchDataConnect);
    synapseIndices.push_back(synapseIndicesConnect);
    strChannelTypeOnSynapse.push_back(type);
  }
#ifdef IDEA_ILEAK
  else if (CG_inAttrPset->identifier == "BRANCH" or
      CG_inAttrPset->identifier == "JUNCTION")
  {
    V.push_back(V_connect);
  }
#endif
}

AnyCurrentDisplay::AnyCurrentDisplay() : CG_AnyCurrentDisplay(), outFile(0) {}

AnyCurrentDisplay::~AnyCurrentDisplay() { delete outFile; }

void AnyCurrentDisplay::duplicate(std::auto_ptr<AnyCurrentDisplay>& dup) const
{
  dup.reset(new AnyCurrentDisplay(*this));
}

void AnyCurrentDisplay::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new AnyCurrentDisplay(*this));
}

void AnyCurrentDisplay::duplicate(
    std::auto_ptr<CG_AnyCurrentDisplay>& dup) const
{
  dup.reset(new AnyCurrentDisplay(*this));
}
