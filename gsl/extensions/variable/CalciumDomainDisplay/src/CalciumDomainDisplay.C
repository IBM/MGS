#include "CG_CalciumDomainDisplay.h"
#include "CalciumDomainDisplay.h"
#include "Lens.h"
#include "MaxComputeOrder.h"
#include <memory>

#define DISTANCE_SQUARED(a, b)                                                 \
  ((((a).x - (b).x) * ((a).x - (b).x)) + (((a).y - (b).y) * ((a).y - (b).y)) + \
   (((a).z - (b).z) * ((a).z - (b).z)))

#define decimal_places 5
#define fieldDelimiter "\t"

void CalciumDomainDisplay::initialize(RNG& rng)
{
  if (Ca.size() > 0)
  {
    if (not deltaT)
    {
      std::cerr << "Please connect time-step to " << typeid(*this).name();
      assert(deltaT);
    }
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
    (*outFile) << "#Time" << fieldDelimiter << "CalciumDomain :";
    if (indices.size() == 0)
    {
      for (unsigned int i = 0; i < dimensions.size(); ++i)
      {
        for (unsigned int j = 0; j < dimensions[i]->size(); ++j)
        {
          // TUAN TODO: potential BUG when key is not unsigned long long
          (*outFile) << std::fixed << " ["
                     << *(reinterpret_cast<unsigned long long*>(
                            &branchData[i]->key))
                     << "," << j << "](" << (*dimensions[i])[j]->x << ","
                     << (*dimensions[i])[j]->y << "," << (*dimensions[i])[j]->z
                     << "," << (*dimensions[i])[j]->r << ","
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
                            &branchData[i]->key))
                     << "," << *it2 << "](" << (*dimensions[i])[*it2]->x << ","
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


void CalciumDomainDisplay::finalize(RNG& rng) {
  if (Ca.size() > 0) outFile->close();
}

void CalciumDomainDisplay::dataCollection(Trigger* trigger,
                                          NDPairList* ndPairList)
{
  if (Ca.size() > 0)
  {
    (*outFile) << float(getSimulation().getIteration()) * *deltaT;
    if (indices.size() == 0)
    {//i.e. record all compartments
      for (auto  ii=0; ii < Ca.size(); ii++)
      {
        ShallowArray<dyn_var_t>* Ca_alldomains = Ca[ii];
        int numCpts = dimensions[ii]->size();
        int tmp = Ca_alldomains->size();
        for (auto jj = offset[ii]; jj < offset[ii] + numCpts; jj++)
          (*outFile) << std::fixed << fieldDelimiter << (*Ca_alldomains)[jj];
      }
      //ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = Ca.begin(),
      //                                                 end1 = Ca.end();
      //int ii=0;
      //for (; it1 != end1; ++it1, ++ii)
      //{
      //  ShallowArray<dyn_var_t>* Ca_alldomains = (*it1);
      //  int numCpts = dimensions[ii]->size();
      //  for (auto jj = offset[ii]; jj < offset[ii] + numCpts; jj++)
      //    (*outFile) << std::fixed << fieldDelimiter << (*Ca_alldomains)[jj];

      //  //ShallowArray<dyn_var_t>::iterator it2 = (*it1)->begin(),
      //  //                                  end2 = (*it1)->end();
      //  //for (; it2 != end2; ++it2)
      //  //  (*outFile) << std::fixed << fieldDelimiter << (*it2);
      //}
    }
    else
    {
      ShallowArray<ShallowArray<dyn_var_t>*>::iterator it1 = Ca.begin(),
                                                       end1 = Ca.end();
      ShallowArray<ShallowArray<int> >::iterator it2 = indices.begin(),
                                                 end2 = indices.end();
      int ii=0;
      for (; it1 != end1; ++it1, ++it2, ii++)
      {
        assert(it2 != end2);
        ShallowArray<dyn_var_t>* Ca_alldomains = (*it1);
        //int numCpts = dimensions[ii]->size();
        ShallowArray<int>::iterator it3 = it2->begin(), end3 = it2->end();
        for (; it3 != end3; ++it3)
        {
          (*outFile) << std::fixed << fieldDelimiter << (*Ca_alldomains)[offset[ii] + (*it3)];
        }
        //ShallowArray<int>::iterator it3 = it2->begin(), end3 = it2->end();
        //for (; it3 != end3; ++it3)
        //{
        //  assert(*it3 < (*it1)->size());
        //  (*outFile) << std::fixed << fieldDelimiter << (**it1)[*it3];
        //}
      }
    }
    (*outFile) << "\n";
  }
}

void CalciumDomainDisplay::setUpPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CalciumDomainDisplayInAttrPSet* CG_inAttrPset,
    CG_CalciumDomainDisplayOutAttrPSet* CG_outAttrPset)
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
    for (unsigned int i = 0; i < dimensions_connect->size(); ++i)
      ind.push_back(i);
  }

  String domain_name = CG_inAttrPset->domainName;
  int index_domain = 0;
  bool domainFound = false;
  for (; index_domain < tmp_microdomainNames->size(); index_domain++)
  {
    if (domain_name == (*tmp_microdomainNames)[index_domain])
    {
      domainFound = true;
      break;
    }
  }
  if (record and domainFound)
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

    int numCpts = dimensions_connect->size();
    offset.push_back(index_domain * numCpts);
  }
}
CalciumDomainDisplay::CalciumDomainDisplay() : CG_CalciumDomainDisplay(), outFile(0) {}
CalciumDomainDisplay::~CalciumDomainDisplay() { delete outFile; }
void CalciumDomainDisplay::duplicate(
    std::unique_ptr<CalciumDomainDisplay>& dup) const
{
  dup.reset(new CalciumDomainDisplay(*this));
}
void CalciumDomainDisplay::duplicate(std::unique_ptr<Variable>& dup) const
{
  dup.reset(new CalciumDomainDisplay(*this));
}
void CalciumDomainDisplay::duplicate(
    std::unique_ptr<CG_CalciumDomainDisplay>& dup) const
{
  dup.reset(new CalciumDomainDisplay(*this));
}
