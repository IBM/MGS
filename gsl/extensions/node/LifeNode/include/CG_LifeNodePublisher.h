// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CG_LifeNodePublisher_H
#define CG_LifeNodePublisher_H

#include "Lens.h"
#include "GeneratedPublisherBase.h"
#include "GenericService.h"
#include "Publishable.h"
#include "ServiceDescriptor.h"
#include "ShallowArray.h"
#include "Simulation.h"
#include <memory>

class CG_LifeNode;
#if defined(HAVE_GPU) && defined(__NVCC__)
class CG_LifeNodeCompCategory;
#endif

class CG_LifeNodePublisher : public GeneratedPublisherBase
{
   public:
      CG_LifeNodePublisher(Simulation& sim, CG_LifeNode* data);
      virtual const std::vector<ServiceDescriptor>& getServiceDescriptors() const;
      virtual std::string getName() const;
      virtual std::string getDescription() const;
      virtual void duplicate(std::unique_ptr<Publisher>& dup) const;
      virtual ~CG_LifeNodePublisher();
   protected:
      virtual Service* createService(const std::string& serviceRequested);
      virtual Service* createOptionalService(const std::string& serviceRequested);
      virtual std::string getServiceNameWithInterface(const std::string& interfaceName, const std::string& subInterfaceName);
   private:
      CG_LifeNode* _data;
      static std::vector<ServiceDescriptor> _serviceDescriptors;
};

#endif
