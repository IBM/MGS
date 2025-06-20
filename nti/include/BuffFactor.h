// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
