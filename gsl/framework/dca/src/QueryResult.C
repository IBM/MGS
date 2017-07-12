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

#include "QueryResult.h"
#include "Queriable.h"

QueryResult::QueryResult()
{
   _numFound = 0;
   _searchCompleted = true;
}


void QueryResult::addQueriable(std::auto_ptr<Queriable> & q)
{
   this->push_back(q.release());
}


QueryResult::~QueryResult()
{
   std::vector<Queriable*>::iterator end = this->end();
   for (std::vector<Queriable*>::iterator iter = begin(); iter != end; iter++)
      delete (*iter);
}
