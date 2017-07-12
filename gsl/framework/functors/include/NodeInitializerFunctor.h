// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef NODEINITIALIZERFUNCTOR_H
#define NODEINITIALIZERFUNCTOR_H
#include "Copyright.h"

#include "Functor.h"
class NodeInitializerFunctor : public Functor {
  public:
  virtual const std::string& getCategory() const;
  static const std::string _category;
};
#endif
