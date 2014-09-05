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
// ================================================================

#ifndef TABLEENTRY_H
#define TABLEENTRY_H

class TableEntry
{
 public:
  virtual ~TableEntry() {}
  virtual void merge(int thisSize, TableEntry*& thatEntry, int& thatSize) =0;
};
 
#endif

