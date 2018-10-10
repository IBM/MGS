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

#ifndef BROWSER_H
#define BROWSER_H
#include "Copyright.h"

#include "PauseActionable.h"
#include "QueryResult.h"
#include "Simulation.h"
#include "SimulationQueriable.h"
#include "QueryField.h"
#include "EnumEntry.h"
#include "Publisher.h"
#include "TextUserInterface.h"
#include "Service.h"
#include "ServiceDescriptor.h"

#include <string>
#include <vector>

class Repertoire;
class TextUserInterface;
class Publisher;
class Queriable;
class Simulation;

class Browser : public PauseActionable
{
   public:
      Browser(Simulation& sim, TextUserInterface* UI);
      void browse();
      void action();
      ~Browser();
   private:
      Simulation& _sim;
      std::string addColon(std::string);
      std::string bracket(std::string, int);
      std::string bracket(int, int);
      std::string getCommand();
      int getSelection();
      int getSelection(const std::string &);
      bool isMatch(std::string input, std::string comp);
      Repertoire* _root;
      Queriable* _currentQ;
      std::vector<Queriable*> _historyQ;
      bool _fieldSet;
      QueryResult _result;
      TextUserInterface* _UI;
      int _i;
};


inline Browser::Browser(Simulation& sim, TextUserInterface* UI)
: _sim(sim)
{
   _currentQ = new SimulationQueriable(&sim);
                                 // only this queriable need be deleted in destructor. JK.
   _historyQ.push_back(_currentQ);
   _UI = UI;
   sim.run();
}

inline Browser::~Browser()
{
   delete _historyQ[0];
}

inline void Browser::browse()
{
   _i = 1;

   QueryResult* result;
   std::string cmd;
   int j = 0;
   std::vector<QueryField*>::iterator fields_iter;
   std::vector<QueryField*>::iterator fields_end;
   std::vector<EnumEntry*>::iterator enums_iter;
   std::vector<EnumEntry*>::iterator enums_end;
   std::vector<Queriable*>::iterator queriables_iter;
   std::vector<Queriable*>::iterator queriables_end;
   std::vector<ServiceDescriptor>::const_iterator services_iter, services_end;

   int fields_idx;
   int enums_idx;
   int queriables_idx;
   int services_idx;

   while (_i) {
      j = 0;
      std::vector<QueryField*> & queryFields = _currentQ->getQueryDescriptor().getQueryFields();
      fields_end = queryFields.end();
      fields_idx = 0;
      for (fields_iter = queryFields.begin(); fields_iter != fields_end; fields_iter++) {
         j++;
         std::string desc = addColon((*fields_iter)->getDescription());
         std::string field = bracket((*fields_iter)->getField(),3);
         std::string format = addColon((*fields_iter)->getFormat());
         std::cout<<bracket(j,1)<<(*fields_iter)->getName()<<desc<<format<<"   "<<field<<std::endl;
      }
      std::cout<<std::endl;
      cmd = getCommand();

      // must have at least one field set to query
      if (isMatch(cmd,"query") && _currentQ->getQueryDescriptor().isAnyFieldSet()) {
         result = (_currentQ->query(100, 0, 10)).release();
         if (result->size() == 0) {
            std::cout<<std::endl<<"No results."<<std::endl;
         }
         else {
            j = 0;
            std::cout<<std::endl<<"RESULT :"<<std::endl<<std::endl;
            queriables_end = result->end();
            for (queriables_iter = result->begin(); queriables_iter != queriables_end; queriables_iter++) {
               j++;
               std::string type = (*queriables_iter)->getQueriableDescriptor().getType();
               std::string name = addColon((*queriables_iter)->getQueriableDescriptor().getName());
               std::string desc = addColon((*queriables_iter)->getQueriableDescriptor().getDescription());
               std::cout<<bracket(j,3)<<type<<name<<desc<<std::endl;
            }
            std::cout<<std::endl;
            if (j == 1) queriables_idx = 1;
            else queriables_idx = getSelection();

            if ((queriables_idx>0) && (queriables_idx<j+1)) {
               queriables_idx-=1;
	       std::unique_ptr<Queriable> dup;
	       ((*result)[queriables_idx])->duplicate(dup);
               _currentQ = dup.release();
               _historyQ.push_back(_currentQ);
               if (_currentQ->isPublisherQueriable()) {
                  std::cout<<"Get Publisher?"<<std::endl;
                  cmd = getCommand();
                  std::cout<<std::endl;
                  if (isMatch(cmd, "yes")) {
                     Publisher* pub = _currentQ->getQPublisher();
                     std::string name=pub->getName();
                     std::string desc = addColon(pub->getDescription());
                     std::cout<<name<<desc<<" : "<<std::endl<<std::endl;
                     const std::vector<ServiceDescriptor>& serviceDescriptors = pub->getServiceDescriptors();
                     j = 0;
                     services_end = serviceDescriptors.end();
                     for (services_iter = serviceDescriptors.begin(); services_iter != services_end; ++services_iter) {
                        j++;
                        std::string name = services_iter->getName();
                        std::string desc = addColon(services_iter->getDescription());
                        std::cout<<bracket(j,4)<<name<<desc<<std::endl;
                     }
                     std::cout<<std::endl;
                     if (j == 1) services_idx = 1;
                     else services_idx = getSelection();
                     if ((services_idx>0) && (services_idx<j+1)) {
                        services_idx-=1;
                        std::cout<<"Service provides a "<<serviceDescriptors[services_idx].getDataItemDescription()<<"."<<std::endl;
                     }
                  }
               }
            }
         }
      }
      else if (isMatch(cmd, "back")) {
         if (_historyQ.size()>1) {
            delete _currentQ;
            _historyQ.pop_back();
            _currentQ = _historyQ[_historyQ.size()-1];
         }
      }
      else if (isMatch(cmd,"clear")) {
         _currentQ->getQueryDescriptor().clearFields();
      }
      else if ((cmd == "cc") || (cmd == "CC")) {
         _currentQ->getQueryDescriptor().clearFields();
         std::vector<Queriable*>::iterator end = _historyQ.end();
         for (std::vector<Queriable*>::iterator iter = _historyQ.begin(); iter != end; iter++)
            (*iter)->getQueryDescriptor().clearFields();
      }
      else if (isMatch(cmd, "history")) {
	std::vector<Queriable*>::iterator iter, end=_historyQ.end();
	std::cout<<std::endl<<":";
	for (iter=_historyQ.begin(); iter!=end; ++iter) {
	  std::vector<QueryField*>& qf=(*iter)->getQueryDescriptor().getQueryFields();
	  if (qf.size()>1) {
	    std::cout<<"[";
	    std::vector<QueryField*>::iterator iter2, end2=qf.end();
	    for (iter2=qf.begin(); iter2!=end2; ++iter2) {
	      if (iter2!=qf.begin()) std::cout<<"|";
	      std::cout<<(*iter2)->getField();
	    }
	    std::cout<<"]:";
	  }
	  else std::cout<<(*(qf.begin()))->getField()<<":";
	}
	std::cout<<std::endl;
      }
      else fields_idx = getSelection(cmd);
      if ((fields_idx>0) && (fields_idx<j+1)) {
         fields_idx-=1;
         // special case, selection means yes
         if ((queryFields[fields_idx]->getName() == "Grid") && (queryFields[fields_idx]->getField()==""))
            queryFields[fields_idx]->setField("Grid");
         else if (queryFields[fields_idx]->getType()==QueryField::ENUM) {
            std::vector<EnumEntry*> enumEntries = queryFields[fields_idx]->getEnumEntries();
            std::cout<<std::endl;
            j = 0;
            enums_end = enumEntries.end();
            for (enums_iter = enumEntries.begin(); enums_iter != enums_end; enums_iter++) {
               j++;
               std::string desc = addColon((*enums_iter)->getDescription());
               std::string value = (*enums_iter)->getValue();
               std::cout<<bracket(j,2)<<value<<desc<<std::endl;
            }
            std::cout<<std::endl;
            enums_idx = getSelection();
            if ((enums_idx>0) && (enums_idx<j+1)) {
               enums_idx-=1;
               queryFields[fields_idx]->setField(enumEntries[enums_idx]->getValue());
            }
         }
         else {
            std::cout<<std::endl<<"Enter "<<queryFields[fields_idx]->getName()<<": "<<std::endl;
            cmd = getCommand();
            queryFields[fields_idx]->setField(cmd);
         }
      }
      std::cout<<std::endl;
   }
}

inline std::string Browser::addColon(std::string str)
{
   if (str!="") str = " : "+str;
   return str;
}

inline std::string Browser::bracket(std::string str, int i)
{
   if (str!="") {
      if (i==1) str = "["+str+"] ";
      if (i==2) str = " <"+str+"> ";
      if (i==3) str = "  ("+str+") ";
      if (i==4) str = "   {"+str+"} ";
   }
   return str;
}

inline std::string Browser::bracket(int j, int i)
{
   std::ostringstream ostr;
   ostr<<j;
   std::string str = ostr.str();
   if (i==1) str = "["+str+"] ";
   if (i==2) str = " <"+str+"> ";
   if (i==3) str = "  ("+str+") ";
   if (i==4) str = "   {"+str+"} ";
   return str;
}

inline std::string Browser::getCommand()
{
   std::string cmd = "PAUSE";
   while ((cmd == "PAUSE") || (cmd == "pause")) cmd = _UI->getCommand();
   if ((cmd == "RESUME") || (cmd == "resume") || (cmd == "QUIT") 
       || (cmd == "quit")) {
      _i = 0;
      std::cout<<std::endl;
   }
   return cmd;
}

inline int Browser::getSelection()
{
   int rval = 0;
   std::string cmd = getCommand();
   std::istringstream istr(cmd);
   istr>>rval;
   if ( !rval ) {
      std::cerr<<"Bad input. Try again."<<std::endl;
      rval = 0;
   }
   return rval;
}

inline int Browser::getSelection(const std::string & cmd)
{
   int rval = 0;
   std::istringstream istr(cmd);
   istr>>rval;
   if ( !rval ) {
      if ((cmd != "RESUME") && (cmd != "resume") && (cmd != "QUIT") 
	  && (cmd != "quit"))
         std::cerr<<"Bad input. Try again."<<std::endl;
      rval = 0;
   }
   return rval;
}

inline bool Browser::isMatch(std::string input, std::string comp)
{
   bool rval = false;
   std::string::iterator inp_iter = input.begin();
   std::string::iterator cmp_iter = comp.begin();
   std::string::iterator cmp_end;

   if (input == comp) rval = true;
   else if (((*inp_iter) == (*cmp_iter)) && (input.length()==1)) rval = true;
   else if (((*inp_iter)+32 == (*cmp_iter)) && (input.length()==1)) rval = true;
   else {
      rval = true;
      if (input.length() == comp.length()) {
         inp_iter = input.begin();
         cmp_end = comp.end();
         for (cmp_iter = comp.begin(); cmp_iter != cmp_end; cmp_iter++) {
            if (((*inp_iter) != (*cmp_iter)) && ((*inp_iter)+32 != (*cmp_iter))) rval = false;
            inp_iter++;
         }
      }
      else rval = false;
   }
   return rval;
}

inline void Browser::action()
{
   browse();
}
#endif
