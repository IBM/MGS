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

#ifndef MemPattern_H
#define MemPattern_H
#include "Copyright.h"

class MemPattern {

 public:

  MemPattern() : orig(0), origDispls(0), origDisplsEnd(0), pattern(0), patternEnd(0) {}
  char* orig;      // byte origin of marshalling or demarshalling
  int* origDispls; // byte offsets at which to apply pattern
  int* origDisplsEnd;
  int* pattern;    // pattern consisting of alternating byte displacements and num bytes
  int* patternEnd;
  
  int* allocateOrigDispls(int nDispls) {
    if (origDispls) delete [] origDispls;
    origDispls = new int[nDispls];
    origDisplsEnd = origDispls+nDispls;
    return origDispls;
  }

  int* allocatePattern(int nBlocks) {
    if (pattern) delete [] pattern;
    int n=nBlocks*2;
    pattern = new int[n];
    patternEnd = pattern+n;
    return pattern;
  }

  ~MemPattern() {
    if (origDispls) delete [] origDispls;
    if (pattern) delete [] pattern;
  }
};

class MemPatternPointers {
 public:
  MemPatternPointers() : _memPatterns(0), _memPatternsEnd(0) {}
  MemPattern *_memPatterns;
  MemPattern* _memPatternsEnd;
};
#endif
