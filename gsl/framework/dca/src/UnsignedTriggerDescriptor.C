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

#include "UnsignedTriggerDescriptor.h"
#include "UnsignedTrigger.h"
#include "UnsignedIntDataItem.h"
#include "Service.h"
#include "Simulation.h"
#include "StringDataItem.h"
#include "ServiceDataItem.h"
#include "IntDataItem.h"
#include "UnsignedIntDataItem.h"
#include "TriggerDataItem.h"
#include "DataItem.h"
#include "Queriable.h"
#include "DataItemQueriable.h"
#include "NDPairList.h"
#include "NDPairItemFinder.h"
#include "NDPairTypeChecker.h"
#include "InstanceFactoryQueriable.h"
#include "SyntaxErrorException.h"

#include <algorithm>
//#include <iostream>
//#include <sstream>

UnsignedTriggerDescriptor::UnsignedTriggerDescriptor(Simulation& s)
: TriggerType("UnsignedTrigger", "UnsignedTrigger", 
	      "Can trigger on any unsigned int."), _sim(s)
{
   std::pair<std::string, DataItem*> p;
   std::vector<std::pair<std::string, DataItem*> > v;

   p.first = "Description";
   p.second = new StringDataItem;
   v.push_back(p);

   p.first = "Predicate";
   p.second = new ServiceDataItem;
   v.push_back(p);

   p.first = "Operator :  == < > != <= >=";
   p.second = new StringDataItem;
   v.push_back(p);

   p.first = "Criterion";
   p.second = new UnsignedIntDataItem;
   v.push_back(p);

   p.first = "Delay";
   p.second = new UnsignedIntDataItem;
   v.push_back(p);

   _parameterDescription.push_back(v);
}


void UnsignedTriggerDescriptor::getQueriable(
   std::auto_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   Array<Trigger*>::iterator it, end = _triggerList.end();
   for (it = _triggerList.begin(); it!=end; ++it) {
      Trigger* t = (*it);
      TriggerDataItem* tdi = new TriggerDataItem;
      tdi->setTrigger(t);
      std::auto_ptr<DataItem> apdi(tdi);
      DataItemQueriable* diq = new DataItemQueriable(apdi);
      diq->setName(t->getDescription());
      std::auto_ptr<DataItemQueriable> apq(diq);
      dup->addQueriable(apq);
   }
   dup->setName(_name);
}


Trigger* UnsignedTriggerDescriptor::getTrigger(const std::vector<DataItem*>& args)
{
   Trigger* rval = new UnsignedTrigger(_sim, args);
   _triggerList.push_back(rval);
   TriggerDataItem* tdi = new TriggerDataItem;
   tdi->setTrigger(rval);
   _instances.push_back(tdi);
   return rval;
}


Trigger* UnsignedTriggerDescriptor::getTrigger(NDPairList& ndp)
{
   Trigger* rval = 0;

   NDPairList::iterator end = ndp.end();

   NDPairItemFinder finder;
   NDPairList::iterator itr, itr2, itr3, itr4, itr5; 
   itr = finder.find(ndp, "Description");
   itr2 = finder.find(ndp, "Service");
   itr3 = finder.find(ndp, "Operator");
   itr5 = finder.find(ndp, "Criterion");
   itr5 = finder.find(ndp, "Delay");
   
   if(itr2 == end) {
      throw SyntaxErrorException("Service not found in UnsignedTriggerDescriptor NDPList!");
   }
   else if(itr3 == end) {
      throw SyntaxErrorException("Operator not found in NDPList!");
   }
   else if(itr4 == end) {
      throw SyntaxErrorException("Criterion not found in NDPList!");
   }
   else if(itr5 == end) {
      throw SyntaxErrorException("Delay not found in UnsignedTriggerDescriptor NDPList!");
   }
   else {
      
      bool descriptionPassed = false;
      StringDataItem descriptionDI;

      if (itr == end) {
	 descriptionDI.setString(_description);
      } else {
	 descriptionPassed = true;
      }

      std::vector<DataItem*> args;
      if (descriptionPassed) {
	 args.push_back(&descriptionDI);
      } else {
	 args.push_back((*itr)->getDataItem());
      }
      args.push_back((*itr2)->getDataItem());
      args.push_back((*itr3)->getDataItem());
      args.push_back((*itr4)->getDataItem());
      args.push_back((*itr5)->getDataItem());

      rval = new UnsignedTrigger(_sim, args);
      _triggerList.push_back(rval);
      TriggerDataItem* tdi = new TriggerDataItem;
      tdi->setTrigger(rval);
      _instances.push_back(tdi);
   }
   return rval;
}

void UnsignedTriggerDescriptor::duplicate(
   std::auto_ptr<TriggerType>& dup) const
{
   dup.reset(new UnsignedTriggerDescriptor(*this));
}

UnsignedTriggerDescriptor::~UnsignedTriggerDescriptor()
{
}
