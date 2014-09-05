// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      std::auto_ptr<QueryResult> query(int maxtItem, int minItem, int searchSize);
      QueryDescriptor & getQueryDescriptor();
      Publisher* getQPublisher();
      bool isPublisherQueriable();
      virtual void duplicate(std::auto_ptr<Queriable>& dup) const;
      void getDataItem(std::auto_ptr<DataItem> &);
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
