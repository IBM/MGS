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
      void addQueriable(std::auto_ptr<Queriable> & q);
      ~QueryResult();
};
#endif
