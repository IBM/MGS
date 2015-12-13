#ifndef Connexon_VmCaiCaERCompCategory_H
#define Connexon_VmCaiCaERCompCategory_H

#include "Lens.h"
#include "CG_Connexon_VmCaiCaERCompCategory.h"

class NDPairList;

class Connexon_VmCaiCaERCompCategory : public CG_Connexon_VmCaiCaERCompCategory, public CountableModel
{
   public:
      Connexon_VmCaiCaERCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
	  void count();
};

#endif
