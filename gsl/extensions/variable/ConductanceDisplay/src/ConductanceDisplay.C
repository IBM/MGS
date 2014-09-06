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
#include "ConductanceDisplay.h"
#include "Simulation.h"
#include "CG_ConductanceDisplay.h"
#include <memory>

#define decimal_places 8

void ConductanceDisplay::initialize(RNG& rng) 
{
  if (g_channel.size()>0 || g_synapse.size()>0) {
    assert(deltaT);
    std::ostringstream os;
    os<<fileName<<getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
    (*outFile)<<"#Time\tConductance :";
    if (g_channel.size()>0) {
      if (indices.size()==0) {
	for (int i=0; i<g_channel.size(); ++i) {
	  if (g_channel[i]->size()==0)
	    std::cerr<<"WARNING : No channel branch information available. Consider initializing ConductanceDisplay at a later phase."<<std::endl;
	  for (int j=0; j<g_channel[i]->size(); ++j) {
	    (*outFile)<<std::fixed<<" ["<<*(reinterpret_cast<unsigned long long*>(&channelBranchData[i]->key))<<","<<j<<"] ";
	  }
	}
      }
      else {
	ShallowArray< int >::iterator it, end=indices.end();
	int idx=0;
	for (int i=0; i<g_channel.size(); ++i) {	
	  for (it=indices.begin(); it!=end; ++it) {
	    if (g_channel[i]->size()==0)
	      std::cerr<<"WARNING : No channel branch information available. Consider initializing ConductanceDisplay at a later phase."<<std::endl;
	    if (*it<g_channel[i]->size())
	      (*outFile)<<std::fixed<<" ["<<*(reinterpret_cast<unsigned long long*>(&channelBranchData[*it]->key))<<","<<*it<<"] ";
	  }
	}
      }
    }
    if (g_synapse.size()>0) {
      for (int i=0; i<g_synapse.size(); ++i) {
	assert(synapseBranchData[i]->size()==2);
	assert(synapseIndices[i].size()==2);
	(*outFile)<<std::fixed<<" ["<<*(reinterpret_cast<unsigned long long*>(&( (*synapseBranchData[i])[0]->key)))<<","<<*(synapseIndices[i][0])<<"|"
		  <<*(reinterpret_cast<unsigned long long*>(&( (*synapseBranchData[i])[1]->key)))<<","<<*(synapseIndices[i][1])<<"] ";
      }
    }
    (*outFile)<<"\n";
  }
}

void ConductanceDisplay::finalize(RNG& rng) 
{
  if (outFile) outFile->close();
}

void ConductanceDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (g_channel.size()>0 || g_synapse.size()>0) {
    (*outFile)<<float(getSimulation().getIteration())**deltaT;
    if (g_channel.size()>0) {
      if (indices.size()==0) {
	ShallowArray< ShallowArray <float>* >::iterator it1 = g_channel.begin(),
	  end1 = g_channel.end();
	for (; it1!=end1; ++it1) {
	  ShallowArray< float >::iterator it2 = (*it1)->begin(),
	    end2 = (*it1)->end();
	  assert((*it1)->size()>0);
	  for (; it2!=end2; ++it2)
	    (*outFile)<<std::fixed<<"\t"<<(*it2);
	}
      }
      else {
	ShallowArray< int >::iterator it2, end2=indices.end();
	ShallowArray< ShallowArray <float>* >::iterator it1 = g_channel.begin(),
	  end1 = g_channel.end();
	for (; it1!=end1; ++it1) {
	  for (it2 = indices.begin(); it2!=end2; ++it2) {
	    if (*it2<(*it1)->size())
	      (*outFile)<<std::fixed<<"\t"<<(**it1)[*it2];
	  }
	}
      }
    }
    if (g_synapse.size()>0) {
      ShallowArray<float*>::iterator it = g_synapse.begin(),
	end = g_synapse.end();
      for (; it!=end; ++it) {
	(*outFile)<<std::fixed<<"\t"<<(**it);
      }
    }
    (*outFile)<<"\n";
  }
}

void ConductanceDisplay::setUpPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_ConductanceDisplayInAttrPSet* CG_inAttrPset, CG_ConductanceDisplayOutAttrPSet* CG_outAttrPset) 
{
  if (CG_inAttrPset->identifier=="SYNAPSE") {
    synapseBranchData.push_back(synapseBranchDataConnect);
    synapseIndices.push_back(synapseIndicesConnect);
  }
  else if (CG_inAttrPset->identifier=="CHANNEL") {
    g_channel.push_back(g_channelConnect);
  }
}

ConductanceDisplay::ConductanceDisplay() 
  : CG_ConductanceDisplay(), outFile(0)
{
}

ConductanceDisplay::~ConductanceDisplay() 
{
  delete outFile;
}

void ConductanceDisplay::duplicate(std::auto_ptr<ConductanceDisplay>& dup) const
{
   dup.reset(new ConductanceDisplay(*this));
}

void ConductanceDisplay::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new ConductanceDisplay(*this));
}

void ConductanceDisplay::duplicate(std::auto_ptr<CG_ConductanceDisplay>& dup) const
{
   dup.reset(new ConductanceDisplay(*this));
}

