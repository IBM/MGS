// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef INSTANCEFACTORY_H
#define INSTANCEFACTORY_H
#include "Copyright.h"

#include <vector>
#include <string>
#include <utility>
#include <memory>


class DataItem;
class GslContext;
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
      virtual void getInstance(std::unique_ptr<DataItem> &, 
			       std::vector<DataItem*> const *, 
			       GslContext *) = 0;
      virtual void getInstance(std::unique_ptr<DataItem> &, 
 			       const NDPairList& ndplist,
			       GslContext *) = 0;
      virtual std::string getName() =0;
      virtual std::string getDescription() =0;
      virtual std::vector<DataItem*> const & getInstances() {
	 return _instances;
      }
      virtual void getQueriable(
	 std::unique_ptr<InstanceFactoryQueriable>& dup) =0;
      virtual ~InstanceFactory();
   protected:
      std::vector<std::vector<std::pair<std::string, DataItem*> > > _parameterDescription;
      std::vector<DataItem*> _instances;
   private:
      void copyOwnedHeap(const InstanceFactory& rv);
      void destructOwnedHeap();
};
#endif
