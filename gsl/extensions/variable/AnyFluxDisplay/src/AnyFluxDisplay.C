#include "Lens.h"
#include "AnyFluxDisplay.h"
#include "CG_AnyFluxDisplay.h"
#include "MaxComputeOrder.h"
#include <memory>
#include "TissueSite.h"

#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

#define decimal_places 6
#define fieldDelimiter "\t"
#ifdef IDEA_CURRENTONCOMPT
void AnyFluxDisplay::initialize(RNG& rng) 
{
  if (J_channel.size() > 0 || J_synapse.size() > 0
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
    (*outFile) << "#Time" << fieldDelimiter << "AnyFlux :";
    os2 << "# time, ";
    if (J_channel.size() > 0)
    {  // via channels
      ShallowArray<ShallowArray<int> >::iterator it = indices_channel.begin(), 
        end = indices_channel.end();
      int ii = 0;
      for (; it != end; ++it, ++ii)
      {
        ShallowArray<int>::iterator it2 = (*it).begin(), 
          end2 = (*it).end();
        for (; it2 != end2; ++it2)
        {
            if ((unsigned)*it2 < J_channel[ii]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &channelBranchData[ii]->key)) << "," << *it
                         << "] ";
              os2 << strChannelTypeOnCompt[ii] << ", "; 
        }
      }
    }
    if (J_synapse.size() > 0)
    {  // via ion-permeable receptor at synapse
      if (connexonBranchData.size() > 0)
      {//gap-junction current
        for (unsigned int i = 0; i < J_synapse.size(); ++i)
        {
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &connexonBranchData[i]->key)) << ","
                     << *(connexonIndices[i]) << "] ";
              os2 << strChannelTypeOnSynapse[i] << ", "; 
        }
      }
      else//synaptic-receptor current
      {
        for (unsigned int i = 0; i < J_synapse.size(); ++i)
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
      ShallowArray<ShallowArray<int> >::iterator it = indices_leak.begin(), 
        end = indices_leak.end();
      int ii = 0;
      for (; it != end; ++it, ++ii)
      {
        ShallowArray<int>::iterator it2 = (*it).begin(), 
          end2 = (*it).end();
        for (; it2 != end2; ++it2)
        {
            if ((unsigned)*it2 < V[ii]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &leakBranchData[ii]->key)) << "," << *it
                         << "] ";
                     //<< "("
                     //<< (*dimensions[i])[j]->x << "," << (*dimensions[i])[j]->y
                     //<< "," << (*dimensions[i])[j]->z << ","
                     //<< (*dimensions[i])[j]->r << ","
                     //<< (*dimensions[i])[j]->dist2soma << ") ";
              os2 << "leak" << ", "; 
        }
      }
    }
#endif
    (*outFile) << "\n";
    (*outFile) << os2.str() << "\n";
  }
}
#else
void AnyFluxDisplay::initialize(RNG& rng)
{
  if (J_channel.size() > 0 || J_synapse.size() > 0
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
    (*outFile) << "#Time" << fieldDelimiter << "AnyFlux :";
    os2 << "# time, ";
    if (J_channel.size() > 0)
    {  // via channels
      if (indices.size() == 0)
      {
        for (unsigned int i = 0; i < J_channel.size(); ++i)
        {
          assert(J_channel[i]->size() > 0);
          for (unsigned int j = 0; j < J_channel[i]->size(); ++j)
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
        for (unsigned int i = 0; i < J_channel.size(); ++i)
        {
          for (it = indices.begin(); it != end; ++it)
          {
            if ((unsigned)*it < J_channel[i]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &channelBranchData[*it]->key)) << "," << *it
                         << "] ";
              os2 << strChannelTypeOnCompt[i] << ", "; 
          }
        }
      }
    }
    if (J_synapse.size() > 0)
    {  // via ion-permeable receptor at synapse
      if (connexonBranchData.size() > 0)
      {
        for (unsigned int i = 0; i < J_synapse.size(); ++i)
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
        for (unsigned int i = 0; i < J_synapse.size(); ++i)
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
#endif

void AnyFluxDisplay::finalize(RNG& rng) 
{
  if (outFile) outFile->close();
}

#ifdef IDEA_CURRENTONCOMPT
void AnyFluxDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (J_channel.size() > 0 || J_synapse.size() > 0
#ifdef IDEA_ILEAK
      || V.size() > 0
#endif
      )
  {
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (J_channel.size() > 0)
    {
      ShallowArray<ShallowArray<int> >::iterator it = indices_channel.begin(), 
        end = indices_channel.end();
      int ii = 0;
      for (; it != end; ++it, ++ii)
      {
        ShallowArray<int>::iterator it2 = (*it).begin(), 
          end2 = (*it).end();
        for (; it2 != end2; ++it2)
        {
            if ((unsigned)*it2 < J_channel[ii]->size())
              (*outFile) << std::fixed << fieldDelimiter << (*J_channel[ii])[*it2];
        }
      }
    }
    if (J_synapse.size() > 0)
    {
      ShallowArray<dyn_var_t*>::iterator it = J_synapse.begin(),
                                         end = J_synapse.end();
      for (; it != end; ++it)
      {
        (*outFile) << std::fixed << fieldDelimiter << (**it);
      }
    }
#ifdef IDEA_ILEAK
    if (V.size() > 0)
    {
      ShallowArray<ShallowArray<int> >::iterator it = indices_leak.begin(), 
        end = indices_leak.end();
      int ii = 0;
      for (; it != end; ++it, ++ii)
      {
        ShallowArray<int>::iterator it2 = (*it).begin(), 
          end2 = (*it).end();
        for (; it2 != end2; ++it2)
        {
            if ((unsigned)*it2 < V[ii]->size())
            {
              float Ileak = *(gLeak[ii]) * ((*V[ii])[*it2] - *(Eleak[ii]));
              (*outFile) << std::fixed << fieldDelimiter << Ileak ;
            }
        }
      }
    }
#endif
    (*outFile) << "\n";
  }
}
#else
void AnyFluxDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList)
{
  if (J_channel.size() > 0 || J_synapse.size() > 0
#ifdef IDEA_ILEAK
      || V.size() > 0
#endif
      )
  {
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (J_channel.size() > 0)
    {
      if (indices.size() == 0)
      {
        ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 =
                                                             J_channel.begin(),
                                                         end1 = J_channel.end();
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
                                                             J_channel.begin(),
                                                         end1 = J_channel.end();
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
    if (J_synapse.size() > 0)
    {
      ShallowArray<dyn_var_t*>::iterator it = J_synapse.begin(),
                                         end = J_synapse.end();
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
#endif


#ifdef IDEA_CURRENTONCOMPT
void AnyFluxDisplay::setUpPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AnyFluxDisplayInAttrPSet* CG_inAttrPset, CG_AnyFluxDisplayOutAttrPSet* CG_outAttrPset) 
{
  TissueSite& site = CG_inAttrPset->site;
  bool record = true;
  ShallowArray<int> ind;
  {//get the indices 'ind'
  if (site.r != 0)
  {//identify the indices of compartments inside a branch 
    //to record based on distance criteria to a site
    if (dimensions_connect == 0)
    {
      std::cerr << "ERROR: The incoming connection to AnyFluxDisplay has a site to limit the nodes; but the nodes does not has the interface DimensionArrayProducer " << std::endl;
      assert(0);
    }
    record = false;
    for (unsigned int i = 0; i < dimensions_connect->size(); ++i)
    {//make sure it connect to the 'whatever-data' associated with 
      //the compartment within the spherical range
      if ((site.r * site.r) >=
          DISTANCE_SQUARED(site, *((*dimensions_connect)[i])))
      {
        ind.push_back(i);
        record = true;
      }
    }
  }
  else if (dimensions_connect != 0 and dimensions_connect->size() > 0)
  {// record data from all compartments associated with the given ComputeBranch
    for (unsigned int i = 0; i < dimensions_connect->size(); ++i)
      ind.push_back(i);
  }
  else {
    if (CG_inAttrPset->identifier == "CHANNEL")
    {
      for (unsigned int i = 0; i < channelBranchData[channelBranchData.size()-1]->size; ++i)
        ind.push_back(i);
    }
#ifdef IDEA_ILEAK
    else if (CG_inAttrPset->identifier == "BRANCH" or
        CG_inAttrPset->identifier == "JUNCTION")
    {
      for (unsigned int i = 0; i < leakBranchData[leakBranchData.size()-1]->size; ++i)
        ind.push_back(i);
    }
#endif
  }
  }
  if (not record)
  {
    //std::cerr << "ERROR: The incoming connection to AnyFluxDisplay has a site to limit the nodes; but there is no compartment within the range " << std::endl;
    //assert(0);
    return;
    //TUAN TODO: if we want to enable this, make sure use branchData_connect
  }
  CustomString type(CG_inAttrPset->type);
  if (type == "")
    type = "unknown";
  if (CG_inAttrPset->identifier == "CHANNEL"
#ifdef IDEA_ILEAK
     or CG_inAttrPset->identifier == "BRANCH" or
      CG_inAttrPset->identifier == "JUNCTION"
#endif
   )
  {
    //now check
    if (record)
    {
      if (ind.size() > 0)
      {
        if (CG_inAttrPset->identifier == "CHANNEL")
        {
          indices_channel.push_back(ind);
          J_channel.push_back(J_channelConnect);
          strChannelTypeOnCompt.push_back(type);
        }
#ifdef IDEA_ILEAK
        else if (CG_inAttrPset->identifier == "BRANCH" or
            CG_inAttrPset->identifier == "JUNCTION")
        {
          indices_leak.push_back(ind);
          V.push_back(V_connect);
        }
#endif
      }
      ////
    }
  }
  else if (CG_inAttrPset->identifier == "SYNAPSE")
  {
    synapseBranchData.push_back(synapseBranchDataConnect);
    synapseIndices.push_back(synapseIndicesConnect);
    strChannelTypeOnSynapse.push_back(type);
  }
  dimensions_connect = 0;
}
#else
void AnyFluxDisplay::setUpPointers(
    const CustomString& CG_direction, const CustomString& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_AnyFluxDisplayInAttrPSet* CG_inAttrPset,
    CG_AnyFluxDisplayOutAttrPSet* CG_outAttrPset)
{
  CustomString type(CG_inAttrPset->type);
  if (type == "")
    type = "unknown";
  if (CG_inAttrPset->identifier == "CHANNEL")
  {
    J_channel.push_back(J_channelConnect);
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
#endif


AnyFluxDisplay::AnyFluxDisplay() 
   : CG_AnyFluxDisplay(), outFile(0)
{
}

AnyFluxDisplay::~AnyFluxDisplay() 
{
  delete outFile;
}

void AnyFluxDisplay::duplicate(std::unique_ptr<AnyFluxDisplay>&& dup) const
{
   dup.reset(new AnyFluxDisplay(*this));
}

void AnyFluxDisplay::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new AnyFluxDisplay(*this));
}

void AnyFluxDisplay::duplicate(std::unique_ptr<CG_AnyFluxDisplay>&& dup) const
{
   dup.reset(new AnyFluxDisplay(*this));
}

