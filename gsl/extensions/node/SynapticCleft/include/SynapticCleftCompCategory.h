#ifndef SynapticCleftCompCategory_H
#define SynapticCleftCompCategory_H

#include "Lens.h"
#include "CG_SynapticCleftCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SynapticCleftCompCategory : public CG_SynapticCleftCompCategory,
                                  public CountableModel
{
  public:
  SynapticCleftCompCategory(Simulation& sim, const std::string& modelName,
                            const NDPairList& ndpList);
  void count();
};

#endif
