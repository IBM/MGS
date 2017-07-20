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

#ifndef RELATIVENODESETDATAITEM_H
#define RELATIVENODESETDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class C_relative_nodeset;

class RelativeNodeSetDataItem : public DataItem
{
   private:
      C_relative_nodeset *_relativeNodeset;

   public:
      static const char* _type;

      virtual RelativeNodeSetDataItem& operator=(const RelativeNodeSetDataItem& DI);

      // Constructors
      RelativeNodeSetDataItem();
      RelativeNodeSetDataItem(std::auto_ptr<C_relative_nodeset> relativeNodeset);
      RelativeNodeSetDataItem(const RelativeNodeSetDataItem& DI);

      // Destructor
      ~RelativeNodeSetDataItem();

      // Utility methods
      void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      C_relative_nodeset* getRelativeNodeSet() const;
      void setRelativeNodeSet(C_relative_nodeset* rns);

};
#endif
