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

#include "DataItemQueriable.h"
#include "DataItem.h"

//#include <sstream>
//#include <iostream>

DataItemQueriable::DataItemQueriable(std::auto_ptr<DataItem> & dataItem)
{
   _dataItem = dataItem.release();
   _publisherQueriable = false;
   _queriableName = _dataItem->getType();
   _queriableDescription = "Access this instance of specified type:";
   _queriableType = "Instance";
}


DataItemQueriable::DataItemQueriable(const DataItemQueriable & q)
   : Queriable(q)
{
   std::auto_ptr<DataItem> apdi;
   q._dataItem->duplicate(apdi);
   _dataItem = apdi.release();
}


void DataItemQueriable::setDescription(std::string description)
{
   _queriableDescription = description;
}


void DataItemQueriable::setName(std::string name)
{
   _queriableName = name;
}


void DataItemQueriable::getDataItem(std::auto_ptr<DataItem> & apdi)
{
   _dataItem->duplicate(apdi);
}


std::auto_ptr<QueryResult> DataItemQueriable::query(int maxItem, int minItem, int searchSize)
{
   std::auto_ptr<QueryResult> qr(new QueryResult());
   std::cerr<<"Queries not implemented on DataItemQueriable!"<<std::endl;
   return qr;
}


Publisher* DataItemQueriable::getQPublisher()
{
   return 0;
}


void DataItemQueriable::duplicate(std::auto_ptr<Queriable>& dup) const
{
   dup.reset(new DataItemQueriable(*this));
}


DataItemQueriable::~DataItemQueriable()
{
   delete _dataItem;
}
