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

#ifndef C_index_set_H
#define C_index_set_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_index_entry;
class LensContext;
class SyntaxError;

class C_index_set : public C_production
{
   public:
      C_index_set(const C_index_set&);
      C_index_set(C_index_entry *, SyntaxError *);
      C_index_set(C_index_set *, C_index_entry *, SyntaxError *);
      C_index_set(SyntaxError *);
      virtual C_index_set* duplicate() const;
      std::list<C_index_entry*>* releaseList();
      virtual ~C_index_set();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<C_index_entry *>& getIndexEntryList() const;

   private:
      std::list<C_index_entry *>* _listIndexEntry;
};
#endif
