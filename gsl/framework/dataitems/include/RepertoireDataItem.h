// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
      virtual void duplicate(std::auto_ptr<DataItem> & r_aptr) const;
      virtual const char* getType() const;
      virtual std::string getString(Error* error=0) const;
};
#endif
