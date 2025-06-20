// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "QueryResult.h"
#include "Queriable.h"

QueryResult::QueryResult()
{
   _numFound = 0;
   _searchCompleted = true;
}


void QueryResult::addQueriable(std::unique_ptr<Queriable> & q)
{
   this->push_back(q.release());
}


QueryResult::~QueryResult()
{
   std::vector<Queriable*>::iterator end = this->end();
   for (std::vector<Queriable*>::iterator iter = begin(); iter != end; iter++)
      delete (*iter);
}
