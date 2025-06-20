// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      RelativeNodeSetDataItem(std::unique_ptr<C_relative_nodeset> relativeNodeset);
      RelativeNodeSetDataItem(const RelativeNodeSetDataItem& DI);

      // Destructor
      ~RelativeNodeSetDataItem();

      // Utility methods
      void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      const char* getType() const;

      C_relative_nodeset* getRelativeNodeSet() const;
      void setRelativeNodeSet(C_relative_nodeset* rns);

};
#endif
