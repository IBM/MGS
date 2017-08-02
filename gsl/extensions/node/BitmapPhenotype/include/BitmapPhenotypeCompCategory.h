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
