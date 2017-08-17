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

#ifndef REPERTOIREFACTORYDATAITEM_H
#define REPERTOIREFACTORYDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"
#include <memory>

class RepertoireFactory;

class RepertoireFactoryDataItem : public DataItem
{
   private:
      RepertoireFactory *_data;

   public:
      static const char* _type;

      RepertoireFactoryDataItem & operator=(const RepertoireFactoryDataItem &);
      // Constructors
      RepertoireFactoryDataItem();
      RepertoireFactoryDataItem(std::auto_ptr<RepertoireFactory> data);
      RepertoireFactoryDataItem(const RepertoireFactoryDataItem& DI);
      ~RepertoireFactoryDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      RepertoireFactory* getFactory(Error* error=0) const;
      void setFactory(std::auto_ptr<RepertoireFactory>&, Error* error=0);
};
#endif
