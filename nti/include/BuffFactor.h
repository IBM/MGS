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

#ifndef _BUFFFACTOR_H
#define _BUFFFACTOR_H

#define BUFF_FACTOR 1.05
#define USABLE_BUFF_FACTOR 1.025

static inline
unsigned int getBuffAllocationSize(unsigned int currentBuffSize)
{
  return unsigned(double(currentBuffSize)*BUFF_FACTOR);
}

static inline
unsigned int getUsableBuffSize(unsigned int currentBuffSize)
{
  return unsigned(double(currentBuffSize)*USABLE_BUFF_FACTOR);
}
#endif
