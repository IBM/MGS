// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BitmapPhenotype_H
#define BitmapPhenotype_H

#include "Mgs.h"
#include "CG_BitmapPhenotype.h"
#include "rndm.h"

class BitmapPhenotype : public CG_BitmapPhenotype
{
   friend class BitmapPhenotypeCompCategory;

   public:
      virtual void initialize(RNG& rng);
      virtual void update(RNG& rng);
      virtual ~BitmapPhenotype();

   private:
      char** image;
};

#endif
