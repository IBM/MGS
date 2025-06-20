// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_index_entry_H
#define C_index_entry_H
#include "Copyright.h"

#include "C_production.h"

class GslContext;
class SyntaxError;

class C_index_entry : public C_production
{
   public:
      C_index_entry(const C_index_entry&);
      C_index_entry(int index, SyntaxError *);
      C_index_entry(int from, int to, SyntaxError *);
      C_index_entry(int from, int increment, int to, SyntaxError *);
      virtual ~C_index_entry();
      virtual C_index_entry* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      int getTo() const {
	return _to;
      }
      int getIncrement() const {
	return _increment;
      }
      int getFrom() const {
	return _from;
      }

   private:
      int _from;
      int _increment;
      int _to;
};
#endif
