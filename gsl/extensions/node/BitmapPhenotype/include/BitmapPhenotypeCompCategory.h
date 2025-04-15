// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef BitmapPhenotypeCompCategory_H
#define BitmapPhenotypeCompCategory_H

#include "Lens.h"
#include "CG_BitmapPhenotypeCompCategory.h"

class NDPairList;

class BitmapPhenotypeCompCategory : public CG_BitmapPhenotypeCompCategory
{
   public:
      BitmapPhenotypeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      virtual void initializeShared(RNG& rng);
      virtual void updateShared(RNG& rng);
      ~BitmapPhenotypeCompCategory();

  private:
      char*** _images;
};

#endif
