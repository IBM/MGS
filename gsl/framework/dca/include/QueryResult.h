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
