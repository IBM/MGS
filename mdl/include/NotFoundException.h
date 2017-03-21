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

#ifndef NotFoundException_H
#define NotFoundException_H
#include "Mdl.h"

#include "GeneralException.h"

class NotFoundException : public GeneralException {
   public:
      NotFoundException(const std::string& error);
      ~NotFoundException();        
};

#endif // NotFoundException_H
