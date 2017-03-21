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

#ifndef GRANULEMAPPERDATAITEM_H
#define GRANULEMAPPERDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class GranuleMapper;

class GranuleMapperDataItem : public DataItem
{
   private:
      GranuleMapper *_granuleMapper;

   public:
      static const char* _type;

      virtual GranuleMapperDataItem& operator=(const GranuleMapperDataItem& DI);

      // Constructors
      GranuleMapperDataItem(GranuleMapper *granuleMapper = 0);
      GranuleMapperDataItem(const GranuleMapperDataItem& DI);

      // Destructor
      ~GranuleMapperDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      GranuleMapper* getGranuleMapper() const;
      void setGranuleMapper(GranuleMapper* t);

};
#endif
