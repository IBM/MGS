// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Translation.h"
#include <string>

Translation::Translation() {}

Translation::Translation(const Translation& t) : _index(t._index) {
  std::copy(t._translation, t._translation + 3, _translation);
  // memcpy(_translation, t._translation, sizeof(double)*3);
}

void Translation::setTranslation(double* translation) {
  std::copy(translation, translation + 3, _translation);
  // TUAN: BUG here fixed
  // memcpy(_translation, translation, sizeof(double) * 3);
}

bool Translation::operator==(const Translation& t) const {
  return (_index == t._index);
}

bool Translation::operator<(const Translation& t) const {
  return (_index < t._index);
}

void Translation::operator+=(const Translation& t) {
  _translation[0] += t._translation[0];
  _translation[1] += t._translation[1];
  _translation[2] += t._translation[2];
}

Translation::~Translation() {}
