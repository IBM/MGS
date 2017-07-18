// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#include "Functor.h"
#include <iostream>

const std::string Functor::_category = "FUNCTOR";

Functor::Functor() {}

const std::string& Functor::getCategory() const { return _category; }

void Functor::initialize(LensContext* c, const std::vector<DataItem*>& args) {
  doInitialize(c, args);
}

void Functor::execute(LensContext* c, const std::vector<DataItem*>& args,
                      std::auto_ptr<DataItem>& rvalue) {
  doExecute(c, args, rvalue);
}

Functor::~Functor() {}
