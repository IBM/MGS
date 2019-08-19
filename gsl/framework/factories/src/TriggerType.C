// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "TriggerType.h"
#include "TriggerDataItem.h"
#include "DataItem.h"
#include "Trigger.h"
#include "SyntaxErrorException.h"
#include "NDPairList.h"

TriggerType::TriggerType(const std::string& modelName, 
			 const std::string& name, 
			 const std::string& description)
   : InstanceFactory(), _modelName(modelName), _name(name),
     _description(description)
{
}

// not necessary for now Array handles itself
// TriggerType::TriggerType(const TriggerType& rv)
//    : InstanceFactory(rv), _modelName(rv._modelName), _name(rv._name), 
//      _description(rv._description)
// {
//    copyOwnedHeap(rv);
// }

// TriggerType& TriggerType::operator=(const TriggerType& rv)
// {
//    if (this != &rv) {
//       InstanceFactory::operator=(rv); // this added while commenting out not tested.
//       destructOwnedHeap();
//       copyOwnedHeap(rv);
//       _modelName = rv._modelName;
//       _name = rv._name;
//       _description = rv._description;
//    }
//    return *this;
// }

void TriggerType::getInstance(std::unique_ptr<DataItem> & adi, 
			      std::vector<DataItem*> const * args, 
			      LensContext* c)
{
   TriggerDataItem* di = new TriggerDataItem();
   di->setTrigger(getTrigger(*args));
   adi.reset(di);
}

void TriggerType::getInstance(std::unique_ptr<DataItem> & adi, 
			      const NDPairList& ndplist,
			      LensContext* c)
{
   throw SyntaxErrorException(
      "Triggers can not be instantiated with Name-DataItem pair lists.");
}

TriggerType::~TriggerType() {
   // not necessary for now Array handles itself
   //destructOwnedHeap();
}

// not necessary for now Array handles itself
// void TriggerType::copyOwnedHeap(const TriggerType& rv)
// {
//    if (rv._triggerList.size() > 0) {
//       std::vector<Trigger*>::const_iterator it, end = rv._triggerList.end();
//       for (it = rv._triggerList.begin(); it!=end; ++it) {
// 	 std::unique_ptr<Trigger> dup;
// 	 (*it)->duplicate(dup);
// 	 _triggerList.push_back(dup.release());
//       }
//    }
// }

// void TriggerType::destructOwnedHeap()
// {
//    if (_triggerList.size() > 0) {
//       std::vector<Trigger*>::iterator it, end = _triggerList.end();
//       for (it = _triggerList.begin(); it!=end; ++it) {
// 	 delete (*it);
//       }
//       _triggerList.clear();
//    }
// }
