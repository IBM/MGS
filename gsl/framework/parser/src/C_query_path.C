// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "C_query_path.h"
#include "C_query_list.h"
#include "C_query.h"
#include "C_query_field_entry.h"
#include "C_query_field_entry_list.h"
#include "C_query_field_set.h"
#include "C_repname.h"
#include "Publisher.h"
#include "Queriable.h"
#include "QueryResult.h"
#include "QueryField.h"
#include "QueryDescriptor.h"
#include "RepertoireQueriable.h"
#include "SimulationQueriable.h"
#include "PublisherQueriable.h"
#include "Repertoire.h"
#include "Simulation.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

#include <sstream>

void C_query_path::internalExecute(LensContext *c)
{
   if (_queryList) _queryList->execute(c);
   if (_repName) _repName->execute(c);

   Queriable* currentQ;
   if (_repName) {
      currentQ = new RepertoireQueriable(_repName->getRepertoire());
   }
   else {
      currentQ = new SimulationQueriable(c->sim);
   }

   if (_queryList) {
      std::list<C_query>* qlist;
      qlist = _queryList->getList();

      std::list<C_query>::iterator iter, end = qlist->end();
      int count = 0;
      for(iter = qlist->begin(); iter != end; ++iter) {
         ++count;
         std::vector<QueryField*>& fields = 
	    currentQ->getQueryDescriptor().getQueryFields();
         if (iter->getType() == C_query::_ENTRY) {
            C_query_field_entry* qfe = iter->getEntry();
            setField(fields, qfe->getFieldName(), qfe->getEntry(), 1);
         }
         else if (iter->getType() == C_query::_SET) {
            std::list<C_query_field_entry>* l = 
	       iter->getSet()->getQueryFieldEntryList()->getList();
            int nbrEntries = l->size();
            std::list<C_query_field_entry>::iterator liter, lend = l->end();
            for (liter = l->begin(); liter != lend; ++liter) {
               setField(fields, liter->getFieldName(), liter->getEntry(), 
			nbrEntries);
            }
         }
         else {
	    std::string mes = "unrecognized type found";
	    throwError(mes);
         }
         if (currentQ->query()->size() == 0) {
	    std::ostringstream messtr;
	    messtr << "query path failed on query " << count 
		   << " : no results";
	    std::string mes = messtr.str();
	    throwError(mes);
         }
         else {
            Queriable* tmp;
            tmp = currentQ;
	    std::auto_ptr<Queriable> dup;
	    currentQ->query()->front()->duplicate(dup);
	    currentQ = dup.release();
            delete tmp;
         }
      }
   }
   if (!currentQ->isPublisherQueriable()) {
      std::string mes = "query path terminates at non-publisher queriable";
      throwError(mes);
   }
   _publisher = currentQ->getQPublisher();
   delete currentQ;
}


C_query_path::C_query_path(const C_query_path& rv)
   : C_production(rv), _queryList(0), _repName(0), _publisher(rv._publisher)
{
   if (rv._queryList) {
      _queryList = rv._queryList->duplicate();
   }
   if (rv._repName) {
      _repName = rv._repName->duplicate();
   }
}


C_query_path::C_query_path(SyntaxError * error)
   : C_production(error), _queryList(0), _repName(0), _publisher(0)
{
}


C_query_path::C_query_path(C_query_list* ql, SyntaxError * error)
   : C_production(error), _queryList(ql), _repName(0), _publisher(0)
{
}


C_query_path::C_query_path(C_repname* rn, C_query_list* ql, 
			   SyntaxError * error)
   : C_production(error), _queryList(ql), _repName(rn), _publisher(0)
{
}


C_query_path* C_query_path::duplicate() const
{
   return new C_query_path(*this);
}


void C_query_path::setField(const std::vector<QueryField*>& fields, 
			    const std::string& name, const std::string& entry, 
			    int nbrEntries)
{
   std::vector<QueryField*>::const_iterator iter = fields.begin(),
      end = fields.end();
   int nbrFields = fields.size();
   if (nbrFields < nbrEntries) {
      std::string mes = "extraneous entries found in query field entry set";
      throwError(mes);
   }
   if (name == "") {
      if (nbrFields > nbrEntries) {
	 std::string mes = entry + " entry requires field name specifier";
	 throwError(mes);
      }
      else (*iter)->setField(entry);
   }
   else {
      for (; iter != end; ++iter) {
         QueryField* qf = (*iter);
         if (qf->getName() == name) {
            qf->setField(entry);
            break;
         }
      }
   }
}

C_query_path::~C_query_path()
{
   delete _queryList;
   delete _repName;
}

void C_query_path::checkChildren() 
{
   if (_queryList) {
      _queryList->checkChildren();
      if (_queryList->isError()) {
         setError();
      }
   }
   if (_repName) {
      _repName->checkChildren();
      if (_repName->isError()) {
         setError();
      }
   }
} 

void C_query_path::recursivePrint() 
{
   if (_queryList) {
      _queryList->recursivePrint();
   }
   if (_repName) {
      _repName->recursivePrint();
   }
   printErrorMessage();
} 
