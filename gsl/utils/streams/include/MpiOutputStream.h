// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MpiOutputStream_H
#define MpiOutputStream_H
#include "Copyright.h"
#include <mpi.h>

#include "MpiAsynchSender.h"
#include "SenderOutputStream.h"

typedef SenderOutputStream<MpiAsynchSender> MpiOutputStream;

#endif
