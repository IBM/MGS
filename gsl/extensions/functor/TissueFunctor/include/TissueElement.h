// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef TissueElement_H
#define TissueElement_H

class TissueFunctor;

class TissueElement
{
 public:
  virtual void setTissueFunctor(TissueFunctor*) =0;
};

#endif
