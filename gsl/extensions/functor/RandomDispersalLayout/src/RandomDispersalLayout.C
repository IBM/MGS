#include "Lens.h"
#include "RandomDispersalLayout.h"
#include "CG_RandomDispersalLayoutBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include "Grid.h"
#include "LayerDefinitionContext.h"
#include "Simulation.h"
#include <math.h>
#include <memory>

void RandomDispersalLayout::userInitialize(LensContext* CG_c, int& total)
{
}

ShallowArray< int > RandomDispersalLayout::userExecute(LensContext* CG_c) 
{
  ShallowArray<int> rval;
  Grid* g = CG_c->layerContext->grid;
  int nbrGridPts = g->getNbrGridNodes();
  for (int i=0; i<nbrGridPts; ++i) rval.push_back(0);
  for (int i=0; i<init.total; ++i) {
    int idx=int(drandom(CG_c->sim->getSharedFunctorRandomSeedGenerator())*double(nbrGridPts));
    rval[idx]++;
  }
  return rval;
}

RandomDispersalLayout::RandomDispersalLayout() 
   : CG_RandomDispersalLayoutBase()
{
}

RandomDispersalLayout::~RandomDispersalLayout() 
{
}

void RandomDispersalLayout::duplicate(std::unique_ptr<RandomDispersalLayout>&& dup) const
{
   dup.reset(new RandomDispersalLayout(*this));
}

void RandomDispersalLayout::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new RandomDispersalLayout(*this));
}

void RandomDispersalLayout::duplicate(std::unique_ptr<CG_RandomDispersalLayoutBase>&& dup) const
{
   dup.reset(new RandomDispersalLayout(*this));
}

