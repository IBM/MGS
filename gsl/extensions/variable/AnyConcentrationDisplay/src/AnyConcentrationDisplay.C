#include "Lens.h"
#include "AnyConcentrationDisplay.h"
#include "CG_AnyConcentrationDisplay.h"
#include "MaxComputeOrder.h"
#include <memory>

#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

#define decimal_places 6
#define fieldDelimiter "\t"

void AnyConcentrationDisplay::initialize(RNG& rng) 
{
  if (concentrations.size() > 0 || NT.size() > 0
      )
  {  // concentrations in Branch/Junction or in SynapticCleft
    assert(deltaT);
    std::ostringstream os;
    std::ostringstream os2;
    os << fileName << getSimulation().getRank();
    outFile = new std::ofstream(os.str().c_str());
    outFile->precision(decimal_places);
    (*outFile) << "#Time" << fieldDelimiter << "AnyConcentration:";
    os2 << "# time, ";
    if (concentrations.size() > 0)
    {  // via channels
      ShallowArray<ShallowArray<int> >::iterator it = indices.begin(), 
        end = indices.end();
      int ii = 0;
      for (; it != end; ++it, ++ii)
      {
        ShallowArray<int>::iterator it2 = (*it).begin(), 
          end2 = (*it).end();
        for (; it2 != end2; ++it2)
        {
            if ((unsigned)*it2 < concentrations[ii]->size())
              (*outFile) << std::fixed << " ["
                         << *(reinterpret_cast<unsigned long long*>(
                                &branchData[ii]->key)) << "," << *it
                         << "] ";
              os2 << strConcentrationTypeOnCompt[ii] << ", "; 
        }
      }
    }
    if (NT.size() > 0)
    {  // via ion-permeable receptor at synapse
      if (cleftBranchData.size() > 0)
      {
        for (unsigned int i = 0; i < NT.size(); ++i)
        {
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &cleftBranchData[i]->key)) << ","
                     << (cleftIndices[i]) << "] ";
              os2 << strConcentrationTypeOnCleft[i] << ", "; 
        }
      }
    }
    (*outFile) << "\n";
    (*outFile) << os2.str() << "\n";
  }
}

void AnyConcentrationDisplay::finalize(RNG& rng) 
{
  if (outFile) outFile->close();
}

void AnyConcentrationDisplay::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (concentrations.size() > 0 || NT.size() > 0
      )
  {  // concentrations in Branch/Junction or in SynapticCleft
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (concentrations.size() > 0)
    {  // via channels
      ShallowArray<ShallowArray<int> >::iterator it = indices.begin(), 
        end = indices.end();
      int ii = 0;
      for (; it != end; ++it, ++ii)
      {
        ShallowArray<int>::iterator it2 = (*it).begin(), 
          end2 = (*it).end();
        for (; it2 != end2; ++it2)
        {
            if ((unsigned)*it2 < concentrations[ii]->size())
              (*outFile) << std::fixed << fieldDelimiter << (*concentrations[ii])[*it2];
        }
      }
    }
    if (NT.size() > 0)
    {  // via ion-permeable receptor at synapse
      ShallowArray<dyn_var_t*>::iterator it = NT.begin(),
        end = NT.end();
      for (; it != end; ++it)
      {
        (*outFile) << std::fixed << fieldDelimiter << (**it);
      }
    }
    (*outFile) << "\n";
  }
}

void AnyConcentrationDisplay::setUpPointers(const CustomString& CG_direction, const CustomString& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_AnyConcentrationDisplayInAttrPSet* CG_inAttrPset, CG_AnyConcentrationDisplayOutAttrPSet* CG_outAttrPset) 
{
  TissueSite& site = CG_inAttrPset->site;
  bool record = true;
  ShallowArray<int> ind;
  {//get the indices 'ind'
    if (site.r != 0)
    {//identify the indices of compartments within the sphere
      //to record based on distance criteria to a site
      if (dimensions_connect == 0)
      {
        std::cerr << "ERROR: The incoming connection to AnyConcentrationDisplay has a site to limit the nodes; but the nodes does not has the interface DimensionArrayProducer " << std::endl;
        assert(0);
      }
      record = false;
      if (CG_inAttrPset->identifier == "BRANCH" or 
          CG_inAttrPset->identifier == "JUNCTION" )
      {
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
      else if (CG_inAttrPset->identifier == "CLEFT")
      {
        if (CG_inAttrPset->side == "post")
        {
          int i = (*cleftIndicesConnect)[1];
          if ((site.r * site.r) >=
              DISTANCE_SQUARED(site, *((*dimensions_connect)[i])))
          {
            ind.push_back(i);
            record = true;
          }

        }else if (CG_inAttrPset->side == "pre" or 
            CG_inAttrPset->side == "" )
        {//default is "pre"
          int i = (*cleftIndicesConnect)[0];
          if ((site.r * site.r) >=
              DISTANCE_SQUARED(site, *((*dimensions_connect)[i])))
          {
            ind.push_back(i);
            record = true;
          }
        }
        else{
          std::cerr << "ERROR: value = " << CG_inAttrPset->side <<
            " --> Unaccept value of InAttribute 'side' " << std::endl;
          assert(0);
        }
      }
      else{
        assert(0);

      }
    }
    else if (dimensions_connect != 0 and dimensions_connect->size() > 0)
    {// record data from all compartments associated with the given ComputeBranch
      if (CG_inAttrPset->identifier == "BRANCH" or 
          CG_inAttrPset->identifier == "JUNCTION" )
      {
        for (unsigned int i = 0; i < dimensions_connect->size(); ++i)
          ind.push_back(i);
      }
      else if (CG_inAttrPset->identifier == "CLEFT")
      {
        if (CG_inAttrPset->side == "post")
        {
          int i = (*cleftIndicesConnect)[1];
          ind.push_back(i);
          record = true;
        }else if (CG_inAttrPset->side == "pre" or 
            CG_inAttrPset->side == "" )
        {//default is "pre"
          int i = (*cleftIndicesConnect)[0];
          ind.push_back(i);
          record = true;
        }
        else{
          std::cerr << "ERROR: value = " << CG_inAttrPset->side <<
            " --> Unaccept value of InAttribute 'side' " << std::endl;
          assert(0);
        }
      }
      else{
        assert(0);

      }
    }
    else {
      if (CG_inAttrPset->identifier == "BRANCH" or
          CG_inAttrPset->identifier == "JUNCTION")
      {
        for (unsigned int i = 0; i < branchData[branchData.size()-1]->size; ++i)
          ind.push_back(i);
      }
      if (CG_inAttrPset->identifier == "CLEFT")
      {
        for (unsigned int i = 0; i < branchData[branchData.size()-1]->size; ++i)
          ind.push_back(i);
      }
    }
  }
  if (not record)
  {
    //std::cerr << "ERROR: The incoming connection to AnyConcentrationDisplay has a site to limit the nodes; but there is no compartment within the range " << std::endl;
    //assert(0);
    return;
    //TUAN TODO: if we want to enable this, make sure use branchData_connect
  }

  //Other settings
  CustomString type(CG_inAttrPset->type);
  if (type == "")
    type = "unknown";
  if (
      CG_inAttrPset->identifier == "BRANCH" or
      CG_inAttrPset->identifier == "JUNCTION"
   )
  {
    //now check
    if (record)
    {
      if (ind.size() > 0)
      {
        indices.push_back(ind);
        concentrations.push_back(concentrations_connect);
        strConcentrationTypeOnCompt.push_back(type);
      }
    }
  }
  else if (CG_inAttrPset->identifier == "CLEFT")
  {
    if (CG_inAttrPset->side == "post")
    {
      cleftBranchData.push_back((*cleftBranchDataConnect)[1]);
    }else if (CG_inAttrPset->side == "pre" or 
        CG_inAttrPset->side == "" )
    {
      cleftBranchData.push_back((*cleftBranchDataConnect)[0]);
    }
    strConcentrationTypeOnCleft.push_back(type);
    cleftIndices = ind;
  }
  dimensions_connect = 0;
}

AnyConcentrationDisplay::AnyConcentrationDisplay() 
   : CG_AnyConcentrationDisplay()
{
}

AnyConcentrationDisplay::~AnyConcentrationDisplay() 
{
}

void AnyConcentrationDisplay::duplicate(std::unique_ptr<AnyConcentrationDisplay>&& dup) const
{
   dup.reset(new AnyConcentrationDisplay(*this));
}

void AnyConcentrationDisplay::duplicate(std::unique_ptr<Variable>&& dup) const
{
   dup.reset(new AnyConcentrationDisplay(*this));
}

void AnyConcentrationDisplay::duplicate(std::unique_ptr<CG_AnyConcentrationDisplay>&& dup) const
{
   dup.reset(new AnyConcentrationDisplay(*this));
}

