// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
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

