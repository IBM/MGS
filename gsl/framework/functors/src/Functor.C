// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Functor.h"
#include <iostream>

const std::string Functor::_category = "FUNCTOR";

Functor::Functor() {}

const std::string& Functor::getCategory() const { return _category; }

void Functor::initialize(LensContext* c, const std::vector<DataItem*>& args) {
  doInitialize(c, args);
}

void Functor::execute(LensContext* c, const std::vector<DataItem*>& args,
                      std::unique_ptr<DataItem>& rvalue) {
  doExecute(c, args, rvalue);
}

Functor::~Functor() {}
