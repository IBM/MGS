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

#ifndef DuplicateException_H
#define DuplicateException_H
#include "Mdl.h"

#include "GeneralException.h"

class DuplicateException : public GeneralException {
   public:
      DuplicateException(const std::string& error);
      ~DuplicateException();        
};

#endif // DuplicateException_H
