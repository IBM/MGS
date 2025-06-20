// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef AccessType_H
#define AccessType_H
#include "Mdl.h"

//namespace AccessType {
//   enum {PUBLIC, PROTECTED, PRIVATE};
//}
enum class AccessType {PUBLIC, PROTECTED, PRIVATE,
   First=PUBLIC, Last=PRIVATE};

#endif
