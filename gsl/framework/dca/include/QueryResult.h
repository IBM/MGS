// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef QUERYRESULT_H
#define QUERYRESULT_H
#include "Copyright.h"

#include <vector>
#include <memory>


class Publisher;
class Queriable;

class QueryResult : public std::vector<Queriable*>
{
   public:
      QueryResult();
      int _numFound;
      int _searchCompleted;
      void addQueriable(std::unique_ptr<Queriable> & q);
      ~QueryResult();
};
#endif
