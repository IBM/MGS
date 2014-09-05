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

#ifndef QUERIABLE_H
#define QUERIABLE_H
#include "Copyright.h"

#include "QueryDescriptor.h"
#include "QueriableDescriptor.h"
#include "QueryResult.h"

#include <list>
#include <string>
#include <vector>
#include <string>
#include <memory>


class QueryField;
class Publisher;
class EnumEntry;
class DataItem;

class Queriable
{
   public:
      Queriable();
      Queriable(const Queriable&);
      virtual std::auto_ptr<QueryResult> query(int maxtItem = 0, int minItem = 0
					  , int searchSize = 0) =0;
      virtual Publisher* getQPublisher() =0;
      virtual void duplicate(std::auto_ptr<Queriable>& dup) const =0;
      virtual void getDataItem(std::auto_ptr<DataItem> &) =0;

      std::list<Queriable*> const & getQueriableList() const;
      QueriableDescriptor & getQueriableDescriptor ();
      QueriableDescriptor & getQueriableDescriptor (std::string context);
      QueryDescriptor & getQueryDescriptor();
      bool isPublisherQueriable();

      virtual ~Queriable();

   protected:
      std::auto_ptr<EnumEntry> & emptyEnum();
                                 // All queriables in this list are created in derived class constructors
      std::list<Queriable*> _queriableList;
      // or added piecemeal.
      // The base class takes care of deleting them however.
      QueryDescriptor _queryDescriptor;
      bool _publisherQueriable;
      QueriableDescriptor _qd;
      std::string _queriableName;
      std::string _queriableDescription;
      std::string _queriableType;

   private:
      std::auto_ptr<EnumEntry> _aptrEmptyEnum;
};
#endif
