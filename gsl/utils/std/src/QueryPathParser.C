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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "QueryPathParser.h"
#include "Queriable.h"
#include "QueryField.h"
#include "QueryResult.h"
#include "EnumEntry.h"
#include "Service.h"
#include "Publisher.h"
#include "Trigger.h"
#include "SimulationQueriable.h"
#include "Simulation.h"

#include <iostream>
#include <sstream>

QueryPathParser::QueryPathParser(Simulation& sim)
: _sim(sim), _publisherQueriable(0)
{
}


Service* QueryPathParser::getService(std::string path)
{
   parsePath(path);
   Service* rval = _publisherQueriable->getQPublisher()->getService(_request);
   delete _publisherQueriable;
   _publisherQueriable = 0;
   return rval;
}


TriggerType* QueryPathParser::getTriggerDescriptor(std::string path)
{
   parsePath(path);
   TriggerType* rval = _publisherQueriable->getQPublisher()->getTriggerDescriptor(_request);
   delete _publisherQueriable;
   _publisherQueriable = 0;
   return rval;
}


Publisher* QueryPathParser::getPublisher(std::string path)
{
   path.append(":Publisher");    // add "publisher" request to path std::string since it is implicit in specification
   parsePath(path);
   Publisher* rval = _publisherQueriable->getQPublisher();
   delete _publisherQueriable;
   _publisherQueriable = 0;
   return rval;
}


void QueryPathParser::parsePath(std::string path)
{
   Queriable* currentQ = new SimulationQueriable(&_sim);
   std::list<std::list <std::string> > queries; // data structure will hold queries after initial parsing

   path.append(":");             // appending this makes parsing easier
   char* chr = const_cast<char*>(path.c_str());
   char* end = chr + path.size();
   for (; chr < end; ++chr) {
      chr = skip(chr);
      std::list<std::string> fields;
      if (*chr == '[') {
         while (*chr != ']' && *chr != '\0') {
            chr = skip(++chr);
            char* p = chr;
            while (*(skip(chr)) != ',' && *(skip(chr)) != ']' && *(skip(chr)) != '\0') ++chr;
            std::string field(p, chr-p);
            fields.push_back(field);
            chr = skip(chr);
         }
         queries.push_back(fields);
         while (*chr != ':' && *chr != '\0') ++chr;
         chr = skip(chr);
      }
      else {
         char* p = chr;
         while (*(skip(chr)) != ':' && *(skip(chr)) != '\0') ++chr;
         std::string field(p, chr-p);
         fields.push_back(field);
         queries.push_back(fields);
         chr = skip(chr);
      }
   }
   // last element should be the request
   std::list<std::string>& request = queries.back();
   if (request.size() != 1) {
      std::cerr<<"Last element of Query Path must be a requested Service or Trigger!"<<std::endl;
      exit(-1);
   }
   _request = request.front();
   queries.pop_back();

   // queries parsed, now excercise queriable interface

   std::list<std::list<std::string> >::iterator queries_iter = queries.begin();
   std::list<std::list<std::string> >::iterator queries_end = queries.end();
   std::list<std::string>::iterator query_iter;
   std::list<std::string>::iterator query_end;
   std::vector<QueryField*>::iterator fields_iter;
   std::vector<QueryField*>::iterator fields_end;
   std::vector<QueryField*>::iterator fields_begin;

   std::list<QueryResult*> results;
   int n = 0;
   for (; queries_iter != queries_end; ++queries_iter) {
      ++n;
      std::list<std::string>& query = (*queries_iter);
      query_end = query.end();
      std::vector<QueryField*> & queryFields = currentQ->getQueryDescriptor().getQueryFields();
      fields_begin = queryFields.begin();
      fields_end = queryFields.end();
      if (queryFields.size() < query.size()) {
         std::cerr<<"Extraneous fields found in Query Path!"<<std::endl;
         exit(-1);
      }
      int frames = queryFields.size()-query.size();
      for (int i = 0; i <= frames; ++i) {
         fields_iter = fields_begin + i;
         for (query_iter = query.begin(); query_iter != query_end; ++query_iter) {
            (*fields_iter)->setField(*query_iter);
            ++fields_iter;
         }
         std::auto_ptr<QueryResult> result(currentQ->query(1,0,1));
         if (result->size() == 0) {
            if (i+1 > frames) {
               std::cerr<<"Query "<<n<<" in query path: \""<<path<<"\""<<std::endl<<"returned no results!"<<std::endl;
               exit(-1);
            }
            currentQ->getQueryDescriptor().clearFields();
         }
         else {
            delete currentQ;
	    std::auto_ptr<Queriable> dup;
	    result->front()->duplicate(dup);
            currentQ = dup.release();
            break;
         }
      }
   }
   if (currentQ->isPublisherQueriable()) _publisherQueriable = currentQ;
   else {
      std::cerr<<"Query Path terminates at non-publishable!"<<std::endl;
      exit(-1);
   }
}


char* QueryPathParser::skip(char* c)
{
   char ch = *c;
   while ( (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\"' || ch == '\'') && *c != '\0') ch = *(++c);
   return c;
}


QueryPathParser::~QueryPathParser()
{
   delete _publisherQueriable;
}
