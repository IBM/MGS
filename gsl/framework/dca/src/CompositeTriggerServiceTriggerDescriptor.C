// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "CompositeTriggerServiceTriggerDescriptor.h"
#include "CompositeTriggerServiceTrigger.h"
#include "Trigger.h"
#include "DataItem.h"
#include "BoolDataItem.h"
#include "CustomStringDataItem.h"
#include "UnsignedIntDataItem.h"
#include "TriggerDataItem.h"
#include "Queriable.h"
#include "DataItemQueriable.h"
#include "InstanceFactoryQueriable.h"
#include "NDPairList.h"
#include "NDPairItemFinder.h"
#include "SyntaxErrorException.h"

#include <iostream>
#include <sstream>

CompositeTriggerServiceTriggerDescriptor::CompositeTriggerServiceTriggerDescriptor(Simulation& s)
   : TriggerType("CompositeTriggerServiceTrigger", "CompositeTriggerServiceTrigger", 
		 "Can trigger on the status of two triggers, A and B."), 
     _sim(s)
{
   std::pair<std::string, DataItem*> p;
   std::vector<std::pair<std::string, DataItem*> > v;

   p.first = "Description";
   p.second = new CustomStringDataItem;
   v.push_back(p);

   p.first = "Trigger A";
   p.second = new TriggerDataItem;
   v.push_back(p);

   p.first = "Criterion A";
   p.second = new BoolDataItem;
   v.push_back(p);

   p.first = "Operator : AND && OR || XOR";
   p.second = new CustomStringDataItem;
   v.push_back(p);

   p.first = "Trigger B";
   p.second = new TriggerDataItem;
   v.push_back(p);

   p.first = "Criterion B";
   p.second = new BoolDataItem;
   v.push_back(p);

   p.first = "Delay";
   p.second = new UnsignedIntDataItem;
   v.push_back(p);

   _parameterDescription.push_back(v);
}


void CompositeTriggerServiceTriggerDescriptor::getQueriable(
   std::unique_ptr<InstanceFactoryQueriable>& dup)
{
   dup.reset(new InstanceFactoryQueriable(this));
   Array<Trigger*>::iterator it, end = _triggerList.end();
   for (it = _triggerList.begin(); it!=end; it++) {
      Trigger* t = (*it);
      TriggerDataItem* tdi = new TriggerDataItem;
      tdi->setTrigger(t);
      std::unique_ptr<DataItem> apdi(tdi);
      DataItemQueriable* diq = new DataItemQueriable(apdi);
      diq->setName(t->getDescription());
      std::unique_ptr<DataItemQueriable> apq(diq);
      dup->addQueriable(apq);
   }
   dup->setName(_name);
}


Trigger* CompositeTriggerServiceTriggerDescriptor::getTrigger(std::vector<DataItem*> const & args)
{
   Trigger* rval = new CompositeTriggerServiceTrigger(_sim, args);
   _triggerList.push_back(rval);
   TriggerDataItem* tdi = new TriggerDataItem;
   tdi->setTrigger(rval);
   _instances.push_back(tdi);
   return rval;
}


Trigger* CompositeTriggerServiceTriggerDescriptor :: getTrigger(NDPairList& ndp)
{
   Trigger* rval = 0;

   NDPairList::iterator end = ndp.end();

   NDPairItemFinder finder;
   NDPairList::iterator itr1, itr2, itr3, itr4, itr5, itr6, itr7; 

   itr1 = finder.find(ndp, "Description");
   itr2 = finder.find(ndp, "TriggerA");
   itr3 = finder.find(ndp, "CriterionA");
   itr4 = finder.find(ndp, "Operator");
   itr5 = finder.find(ndp, "TriggerB");
   itr6 = finder.find(ndp, "CriterionB");
   itr7 = finder.find(ndp, "Delay");

   if(itr2 == end) {
      std::cerr<< "TriggerA not found in NDPList!" << std::endl;
      exit(-1);
   }
   if(itr3 == end) {
      std::cerr<< "CriterionA not found in NDPList!" << std::endl;
      exit(-1);
   }
   if(itr4 == end) {
      std::cerr<< "Operator not found in NDPList!" << std::endl;
      exit(-1);
   }
   if(itr5 == end) {
      std::cerr<< "TriggerB not found in NDPList!" << std::endl;
      exit(-1);
   }
   if(itr6 == end) {
      std::cerr<< "CriterionB not found in NDPList!" << std::endl;
      exit(-1);
   }
   if(itr7 == end) {
      std::cerr<< "Delay not found in NDPList!" << std::endl;
      exit(-1);
   }
   else {
      bool descriptionPassed = false;
      CustomStringDataItem descriptionDI;

      if (itr1 == end) {
	 descriptionDI.setString(_description);
      } else {
	 descriptionPassed = true;
      }

      std::vector<DataItem*> args;
      if (descriptionPassed) {
	 args.push_back(&descriptionDI);
      } else {
	 args.push_back((*itr1)->getDataItem());
      }
      args.push_back((*itr2)->getDataItem());
      args.push_back((*itr3)->getDataItem());
      args.push_back((*itr4)->getDataItem());
      args.push_back((*itr5)->getDataItem());
      args.push_back((*itr6)->getDataItem());
      args.push_back((*itr7)->getDataItem());

      rval = new CompositeTriggerServiceTrigger(_sim, args);
      _triggerList.push_back(rval);
      TriggerDataItem* tdi = new TriggerDataItem;
      tdi->setTrigger(rval);
      _instances.push_back(tdi);
   }
   return rval;
}

void CompositeTriggerServiceTriggerDescriptor::duplicate(
   std::unique_ptr<TriggerType>& dup) const
{
   dup.reset(new CompositeTriggerServiceTriggerDescriptor(*this));
}

CompositeTriggerServiceTriggerDescriptor:: ~CompositeTriggerServiceTriggerDescriptor()
{
}
