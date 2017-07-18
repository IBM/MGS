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

#ifndef C_index_entry_H
#define C_index_entry_H
#include "Copyright.h"

#include "C_production.h"

class LensContext;
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
      virtual void internalExecute(LensContext *);
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
