// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-2012
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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
