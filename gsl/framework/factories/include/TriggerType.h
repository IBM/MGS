// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TRIGGERTYPE_H
#define TRIGGERTYPE_H
#include "Copyright.h"

//#include "Publishable.h"
#include "InstanceFactory.h"
//#include "NDPairList.h"
#include "Trigger.h"
#include "DuplicatePointerArray.h"

#include <vector>
#include <memory>
#include <string>

class ParameterSet;
//class Trigger;
class Publisher;
class NDPairList;

class TriggerType : public InstanceFactory
{
   public:
      virtual Trigger* getTrigger(std::vector<DataItem*> const & args)=0;
      TriggerType(const std::string& modelName = "",const std::string& name = "", 
		  const std::string& description = "");
      // not necessary for now Array handles itself
      // TriggerType(const TriggerType& rv);
      // TriggerType& operator=(const TriggerType& rv);
      virtual void duplicate(std::unique_ptr<TriggerType>& dup) const = 0;
      std::string getModelName() {return _modelName;}
      virtual void getInstance(std::unique_ptr<DataItem> &, 
			       std::vector<DataItem*> const *, 
			       GslContext* c = 0);
      virtual void getInstance(std::unique_ptr<DataItem> & adi, 
			       const NDPairList& ndplist,
			       GslContext* c);
      virtual std::string getName() {return _name;}
      virtual std::string getDescription() {return _description;}
      virtual ~TriggerType();
   protected:
      DuplicatePointerArray<Trigger, 50> _triggerList;
      std::string _modelName;
      std::string _name;
      std::string _description;
   private:
      // not necessary for now Array handles itself
      // void copyOwnedHeap(const TriggerType& rv);
      // void destructOwnedHeap();
};
#endif
