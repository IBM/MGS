// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TABLEENTRY_H
#define TABLEENTRY_H

class TableEntry
{
 public:
  virtual ~TableEntry() {}
  virtual void merge(int thisSize, TableEntry*& thatEntry, int& thatSize) =0;
};
 
#endif

