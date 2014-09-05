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

#ifndef C_query_field_entry_H
#define C_query_field_entry_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class C_ndpair_clause;
class C_constant;
class C_declarator;
class SyntaxError;

class C_query_field_entry : public C_production
{
   public:
      C_query_field_entry(C_query_field_entry const &);
      C_query_field_entry(C_ndpair_clause *, SyntaxError *);
      C_query_field_entry(std::string *, SyntaxError *);
      C_query_field_entry(C_constant *, SyntaxError *);
      C_query_field_entry(C_declarator *, SyntaxError *);
      virtual ~C_query_field_entry();
      virtual C_query_field_entry* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::string getFieldName() {
	 return _fieldName;
      }
      std::string getEntry() {
	 return *_entry;
      }

   private:
      std::string _fieldName;
      std::string* _entry;
      C_ndpair_clause* _ndpClause;
      C_constant* _constant;
      C_declarator* _declarator;
};
#endif
