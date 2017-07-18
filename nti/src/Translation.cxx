// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. and EPFL 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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

bool Translation::operator==(const Translation& t) {
  return (_index == t._index);
}

bool Translation::operator<(const Translation& t) {
  return (_index < t._index);
}

void Translation::operator+=(const Translation& t) {
  _translation[0] += t._translation[0];
  _translation[1] += t._translation[1];
  _translation[2] += t._translation[2];
}

Translation::~Translation() {}
