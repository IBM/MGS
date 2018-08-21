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

#ifndef BitmapPhenotype_H
#define BitmapPhenotype_H

#include "Lens.h"
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
