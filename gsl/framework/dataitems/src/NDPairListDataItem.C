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

#include "NDPairListDataItem.h"
#include "NDPairList.h"
#include "NDPair.h"

// Type
const char* NDPairListDataItem::_type = "NDPAIRLIST";

// Constructors
NDPairListDataItem::NDPairListDataItem()
   : _data(0)
{
}

NDPairListDataItem::NDPairListDataItem(std::auto_ptr<NDPairList> data)
{
   _data = data.release();
}

NDPairListDataItem::NDPairListDataItem(const NDPairListDataItem& DI)
   : _data(0)
{
   copyContents(DI);
}


NDPairListDataItem::~NDPairListDataItem()
{
   destructContents();
}

// Utility methods
void NDPairListDataItem::duplicate(std::auto_ptr<DataItem> & r_aptr) const
{
   r_aptr.reset(static_cast<DataItem*> (new NDPairListDataItem(*this)));
}


NDPairListDataItem& NDPairListDataItem::operator=(const NDPairListDataItem& DI)
{
   if (this == &DI) {
      return *this;
   }
   destructContents();
   copyContents(DI);
   return *this;
}


const char* NDPairListDataItem::getType() const
{
   return _type;
}


NDPairList* NDPairListDataItem::getNDPairList(Error* error) const
{
   return _data;
}

void NDPairListDataItem::releaseNDPairList(std::auto_ptr<NDPairList>& ap, Error* error)
{
   ap.reset(_data);
   _data = 0;
}

void NDPairListDataItem::setNDPairList(std::auto_ptr<NDPairList>& ap, Error* error)
{
   delete _data;
   _data = ap.release();
}

std::string NDPairListDataItem::getString(Error* error) const
{
   std::string retval("< ");
   if (_data) {
      NDPairList::const_iterator it, end = _data->end();
      NDPairList::const_iterator check = _data->begin();
      check++;
      for (it = _data->begin(); it != end; it++, check++) {
	 retval += (*it)->getName() + " = " + (*it)->getValue();
	 if (check != end) {
	    retval += ", ";
	 }
      }
   }
   retval += " >";
   return retval;
}

void NDPairListDataItem::copyContents(const NDPairListDataItem& DI)
{
   if (DI._data) {
      _data = new NDPairList();
      *_data = *(DI._data);
   }
}

void NDPairListDataItem::destructContents()
{
   delete _data;
}
