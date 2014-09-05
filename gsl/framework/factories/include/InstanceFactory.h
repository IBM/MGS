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

#ifndef INSTANCEFACTORY_H
#define INSTANCEFACTORY_H
#include "Copyright.h"

#include <vector>
#include <string>
#include <utility>
#include <memory>


class DataItem;
class LensContext;
class InstanceFactoryQueriable;
class NDPairList;

class InstanceFactory
{
   public:
      InstanceFactory();
      InstanceFactory(const InstanceFactory& rv);
      InstanceFactory& operator=(const InstanceFactory& rv);
      virtual std::vector<std::vector<std::pair<std::string, DataItem*> > > 
      const & getParameterDescription() {
	 return _parameterDescription;
      }
      virtual void getInstance(std::auto_ptr<DataItem> &, 
			       std::vector<DataItem*> const *, 
			       LensContext *) = 0;
      virtual void getInstance(std::auto_ptr<DataItem> &, 
 			       const NDPairList& ndplist,
			       LensContext *) = 0;
      virtual std::string getName() =0;
      virtual std::string getDescription() =0;
      virtual std::vector<DataItem*> const & getInstances() {
	 return _instances;
      }
      virtual void getQueriable(
	 std::auto_ptr<InstanceFactoryQueriable>& dup) =0;
      virtual ~InstanceFactory();
   protected:
      std::vector<std::vector<std::pair<std::string, DataItem*> > > _parameterDescription;
      std::vector<DataItem*> _instances;
   private:
      void copyOwnedHeap(const InstanceFactory& rv);
      void destructOwnedHeap();
};
#endif
