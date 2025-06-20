// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef COMPOSITESWC_H
#define COMPOSITESWC_H

#include <string.h>

class CompositeSwc
{
 public:
  CompositeSwc(const char* tissueFileName, const char* outFileName, double sampleRate, bool multicolor);
   ~CompositeSwc() {}
};

#endif /* COMPOSITESWC_H */
