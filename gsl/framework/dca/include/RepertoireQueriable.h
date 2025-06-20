// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef REPERTOIREQUERIABLE_H
#define REPERTOIREQUERIABLE_H
#include "Copyright.h"

#include "Queriable.h"
#include "QueriableDescriptor.h"

#include <list>
#include <string>
#include <vector>
#include <memory>
#include <map>


class Repertoire;
class GridLayerDescriptor;
class ConnectionSet;
class QueryDescriptor;
class QueryField;
class Publisher;
class QueryResult;

class RepertoireQueriable : public Queriable
{
   public:
      RepertoireQueriable(Repertoire* repertoire);
      RepertoireQueriable(const RepertoireQueriable &);
      std::list<QueriableDescriptor> const & getQueriableList() const;
      QueriableDescriptor & getQueriableDescriptor();
      QueriableDescriptor & getQueriableDescriptor(std::string context);
      std::unique_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      QueryDescriptor & getQueryDescriptor();
      Publisher* getQPublisher();
      bool isPublisherQueriable();
      virtual void duplicate(std::unique_ptr<Queriable>& dup) const;
      void getDataItem(std::unique_ptr<DataItem> &);
      ~RepertoireQueriable();

   private:
      std::list<Repertoire*> _subRepertoires;
      std::map<GridLayerDescriptor*, std::list<ConnectionSet*> > _connectionSetMap;
      Repertoire* _repertoire;
      int _gridIdx;
      int _subRepertoireIdx;
      int _connectionSetPreIdx;
      int _connectionSetPostIdx;
      void setConnectionSetQueryResult (QueryResult*, std::string preName, std::string postName, int maxItem, int minItem, int searchSize);
};
#endif
