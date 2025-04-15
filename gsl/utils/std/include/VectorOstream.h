// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VECTOROSTREAM_H
#define VECTOROSTREAM_H
#include "Copyright.h"

// This overloading of the std::ostream operator for vectors is useful for
// writing LENS data to files.

#include <iostream>
#include <vector>
#include <string>

extern std::ostream & operator<<(std::ostream &os, std::vector<int> &v);
extern std::ostream & operator<<(std::ostream &os, const std::vector<int> &v);

extern std::ostream & operator<<(std::ostream &os, std::vector<unsigned> &v);
extern std::ostream & operator<<(std::ostream &os, const std::vector<unsigned> &v);

extern std::ostream & operator<<(std::ostream &os, std::vector<short> &v);
extern std::ostream & operator<<(std::ostream &os, const std::vector<short> &v);

extern std::ostream & operator<<(std::ostream &os, std::vector<float> &v);
extern std::ostream & operator<<(std::ostream &os, const std::vector<float> &v);

extern std::ostream & operator<<(std::ostream &os, std::vector<double> &v);
extern std::ostream & operator<<(std::ostream &os, const std::vector<double> &v);

extern std::ostream & operator<<(std::ostream &os, std::vector<std::string> &v);
extern std::ostream & operator<<(std::ostream &os, const std::vector<std::string> &v);

#endif
