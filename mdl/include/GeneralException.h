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

#ifndef GeneralException_H
#define GeneralException_H
#include "Mdl.h"

#include <string>

class GeneralException {
   public:
      GeneralException(const std::string& error);
      const std::string& getError() const;
      void setError(const std::string& error);
      ~GeneralException();        

   private:
      std::string _error;
};

#endif // GeneralException_H
