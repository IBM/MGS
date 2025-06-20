// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef REPERTOIREDATAITEM_H
#define REPERTOIREDATAITEM_H
#include "Copyright.h"

#include "DataItem.h"

class Repertoire;

class RepertoireDataItem : public DataItem
{
   protected:
      Repertoire *_repertoire;

   public:
      static const char* _type;

      virtual RepertoireDataItem & operator=(const RepertoireDataItem &);

      // Constructors
      RepertoireDataItem(Repertoire *repertoire = 0);
      RepertoireDataItem(const RepertoireDataItem& DI);

      Repertoire* getRepertoire() const;
      void setRepertoire(Repertoire *);

      // Utility methods
      virtual void duplicate(std::unique_ptr<DataItem> & r_aptr) const;
      virtual const char* getType() const;
      virtual std::string getString(Error* error=0) const;
};
#endif
