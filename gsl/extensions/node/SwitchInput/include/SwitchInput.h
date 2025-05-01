// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SwitchInput_H
#define SwitchInput_H

#include "Mgs.h"
#include "CG_SwitchInput.h"
#include "rndm.h"
#include <fstream>


class SwitchInput : public CG_SwitchInput
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      void outputDrivInp(std::ofstream &);
      virtual ~SwitchInput();

};

#endif
