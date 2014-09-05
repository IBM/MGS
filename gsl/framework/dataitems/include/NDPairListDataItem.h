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

#ifndef NDPAIRLISTDATAITEM_H
#define NDPAIRLISTDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include "NDPairList.h"

#include <list>
#include <memory>

class NDPairListDataItem : public DataItem
{
   public:
      static const char* _type;

      NDPairListDataItem & operator=(const NDPairListDataItem & DI);

      // Constructors
      NDPairListDataItem();
      NDPairListDataItem(std::auto_ptr<NDPairList> data);
      NDPairListDataItem(const NDPairListDataItem& DI);
      
      // Destructor
      ~NDPairListDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      NDPairList* getNDPairList(Error* error=0) const;
      void releaseNDPairList(std::auto_ptr<NDPairList>& ap, Error* error=0);
      void setNDPairList(std::auto_ptr<NDPairList>& ap, Error* error=0);
      virtual std::string getString(Error* error=0) const;
   private:
      NDPairList* _data;
      inline void copyContents(const NDPairListDataItem& DI);
      inline void destructContents();
};
#endif
