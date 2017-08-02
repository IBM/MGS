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

#ifndef MpiOutputStream_H
#define MpiOutputStream_H
#include "Copyright.h"
#include <mpi.h>

#include "MpiAsynchSender.h"
#include "SenderOutputStream.h"

typedef SenderOutputStream<MpiAsynchSender> MpiOutputStream;

#endif
