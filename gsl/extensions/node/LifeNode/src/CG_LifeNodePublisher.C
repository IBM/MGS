// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-12-03-2018
//
//  (C) Copyright IBM Corp. 2005-2018  All rights reserved   .
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Lens.h"
#include "CG_LifeNodePublisher.h"
#include "CG_LifeNode.h"
#include "CG_LifeNodeCompCategory.h"
#include "GeneratedPublisherBase.h"
#include "GenericService.h"
#include "Publishable.h"
#include "ServiceDescriptor.h"
#include "ShallowArray.h"
#include "Simulation.h"
#include <memory>

CG_LifeNodePublisher::CG_LifeNodePublisher(Simulation& sim, CG_LifeNode* data) 
   : GeneratedPublisherBase(sim), _data(data){
   if (_serviceDescriptors.size() == 0) {
      _serviceDescriptors.push_back(ServiceDescriptor("value", "", "int"));
      _serviceDescriptors.push_back(ServiceDescriptor("publicValue", "", "int"));
      _serviceDescriptors.push_back(ServiceDescriptor("neighbors", "", "ShallowArray< int* >"));
      _serviceDescriptors.push_back(ServiceDescriptor("tooCrowded", "", "int"));
      _serviceDescriptors.push_back(ServiceDescriptor("tooSparse", "", "int"));
   }
}

const std::vector<ServiceDescriptor>& CG_LifeNodePublisher::getServiceDescriptors() const
{
   return _serviceDescriptors;
}

std::string CG_LifeNodePublisher::getName() const
{
   return "LifeNodePublisher";
}

std::string CG_LifeNodePublisher::getDescription() const
{
   return "";
}

void CG_LifeNodePublisher::duplicate(std::unique_ptr<Publisher>& dup) const
{
   dup.reset(new CG_LifeNodePublisher(*this));
}

Service* CG_LifeNodePublisher::createService(const std::string& serviceRequested) 
{
   Service* rval = 0;
#ifdef HAVE_GPU
   if (serviceRequested == "value") {
      rval = new GenericService< int >(_data, &((_data->getCompCategory()->um_value)[_data->index]));
      _services.push_back(rval);
      return rval;
   }
#else
   if (serviceRequested == "value") {
      rval = new GenericService< int >(_data, &(_data->value));
      _services.push_back(rval);
      return rval;
   }
#endif
#ifdef HAVE_GPU
   if (serviceRequested == "publicValue") {
      rval = new GenericService< int >(_data, &((_data->getCompCategory()->um_publicValue)[_data->index]));
      _services.push_back(rval);
      return rval;
   }
#else
   if (serviceRequested == "publicValue") {
      rval = new GenericService< int >(_data, &(_data->publicValue));
      _services.push_back(rval);
      return rval;
   }
#endif
#ifdef HAVE_GPU
   if (serviceRequested == "neighbors") {
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
      rval = new GenericService< ShallowArray_Flat< int* > >(_data, &((_data->getCompCategory()->um_neighbors)[_data->index]));
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
      int offset = _data->getCompCategory()->um_neighbors_start_offset[_data->index];
neighbors_num_elements[_data->index];
      rval = new GenericService< < int*  >(_data, &((_data->getCompCategory()->um_neighbors)[offset]));
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b
      int offset = _data->index * _data->getCompCategory()->um_neighbors_max_elements;
      rval = new GenericService< < int*  >(_data, &((_data->getCompCategory()->um_neighbors)[offset]));
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5
      assert(0);
#endif
      _services.push_back(rval);
      return rval;
   }
#else
   if (serviceRequested == "neighbors") {
      rval = new GenericService< ShallowArray< int* > >(_data, &(_data->neighbors));
      _services.push_back(rval);
      return rval;
   }
#endif
#ifdef HAVE_GPU
   if (serviceRequested == "tooCrowded") {
      rval = new GenericService< int >(_data, &((_data->getNonConstSharedMembers().tooCrowded));
      _services.push_back(rval);
      return rval;
   }
#else
   if (serviceRequested == "tooCrowded") {
      rval = new GenericService< int >(_data, &(_data->getNonConstSharedMembers().tooCrowded));
      _services.push_back(rval);
      return rval;
   }
#endif
#ifdef HAVE_GPU
   if (serviceRequested == "tooSparse") {
      rval = new GenericService< int >(_data, &((_data->getNonConstSharedMembers().tooSparse));
      _services.push_back(rval);
      return rval;
   }
#else
   if (serviceRequested == "tooSparse") {
      rval = new GenericService< int >(_data, &(_data->getNonConstSharedMembers().tooSparse));
      _services.push_back(rval);
      return rval;
   }
#endif
   return rval;
}

Service* CG_LifeNodePublisher::createOptionalService(const std::string& serviceRequested) 
{
   Service* rval = 0;
   return rval;
}

std::string CG_LifeNodePublisher::getServiceNameWithInterface(const std::string& interfaceName, const std::string& subInterfaceName) 
{
   if (interfaceName == "ValueProducer") {
      if (subInterfaceName == "value") {
         return "publicValue";
      }
   }
   return "";
}

CG_LifeNodePublisher::~CG_LifeNodePublisher() 
{
}

std::vector<ServiceDescriptor> CG_LifeNodePublisher::_serviceDescriptors;
