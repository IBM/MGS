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

#ifndef C_query_H
#define C_query_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class C_query_field_entry;
class C_query_field_set;
class SyntaxError;

class C_query : public C_production
{
   public:
      enum Type {_ENTRY, _SET};
      C_query(C_query const &);
      C_query(C_query_field_entry *, SyntaxError *);
      C_query(C_query_field_set *, SyntaxError *);
      virtual ~C_query ();
      virtual C_query* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_query_field_entry* getEntry() {
	 return _entry;
      }
      C_query_field_set* getSet() {
	 return _set;
      }
      Type getType() {
	 return _type;
      }

   private:
      C_query_field_entry* _entry;
      C_query_field_set* _set;
      Type _type;
};
#endif
