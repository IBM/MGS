#include "Lens.h"
#include "NormalizedGradientLayout.h"
#include "CG_NormalizedGradientLayoutBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include "VolumeOdometer.h"
#include "Grid.h"
#include "LayerDefinitionContext.h"
#include <vector>
#include <map>
#include "Simulation.h"
#include <memory>
#include <cmath>
#include <algorithm>

void NormalizedGradientLayout::userInitialize(LensContext* CG_c, int& total, double& slope, ShallowArray< int >& origin, ShallowArray< bool >& gradientDimensions) 
{
}

ShallowArray< int > NormalizedGradientLayout::userExecute(LensContext* CG_c) 
{
  if (std::abs(init.slope)>1) {
    std::cerr << "Warning: absolute value of slope argument to NormalizedGardientFunctor is greater 1, this will results in many grid points grid point with zero density" << std::endl;
  }
  ShallowArray<int> rval;
  Grid* g = CG_c->layerContext->grid;
  int nbrGridPts = g->getNbrGridNodes();
  for (int i=0; i<nbrGridPts; ++i) rval.push_back(0);

  std::vector<int> gridSize = g->getSize();
  unsigned dims = g->getDimensions();
  if (init.origin.size() != dims) {
    std::cerr << "Error: origin coordinates used in NormalizedGradientFunctor must use the same number of dimensions as grid" << std::endl;
    abort();
  }
  for (int d=0; d<gridSize.size(); d++) {
    if (init.origin[d] > gridSize[d]) {
      std::cerr << "Error: origin coordinates used in NormalizedGradientFunctor must be within grid size coordinates" << std::endl;
      abort();
    }
  }
  
  float p_bar = (float(init.total)/float(nbrGridPts));
  double max_p = p_bar*(1.0+std::abs(init.slope));
  if (p_bar <= 0) {
      std::cerr << "Error: total number of units in NormalizedGradientFunctor or grid size is incorrect and results in zero or negative ratio of units / grid point --> ratio = " << p_bar << std::endl;
      abort();
  } 

  std::vector<double> dists(nbrGridPts);
  std::vector<int> vobeg(dims,0);
  std::vector<int> voend;
  for (int n=0; n<dims; ++n) voend.push_back(gridSize[n]-1);

  VolumeOdometer vo(vobeg, voend);
  while (!vo.isAtEnd()) {
    std::vector<int> next=vo.look();
    double sumsqs=0;
    for (int n=0; n<dims; ++n) {
      if (init.gradientDimensions[n]) 
	sumsqs  += (next[n]-init.origin[n]) * (next[n]-init.origin[n]);	
    }
    dists[g->getNodeIndex(next)] = sqrt(sumsqs);
    vo.next();
  }
  
  auto max_dist = std::max_element(std::begin(dists), std::end(dists));

  int remaining = init.total;
  while(remaining>0) {
    int idx=int(drandom(CG_c->sim->getSharedFunctorRandomSeedGenerator())*double(nbrGridPts));
    float p = p_bar * (1.0 + init.slope * ((2.0 * dists[idx] / *max_dist) - 1.0));
    if (drandom(CG_c->sim->getSharedFunctorRandomSeedGenerator())*max_p < p) {
      rval[idx]++;
      remaining--;
    }
  }
  return rval;
}

NormalizedGradientLayout::NormalizedGradientLayout() 
   : CG_NormalizedGradientLayoutBase()
{
}

NormalizedGradientLayout::~NormalizedGradientLayout() 
{
}

void NormalizedGradientLayout::duplicate(std::unique_ptr<NormalizedGradientLayout>&& dup) const
{
   dup.reset(new NormalizedGradientLayout(*this));
}

void NormalizedGradientLayout::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new NormalizedGradientLayout(*this));
}

void NormalizedGradientLayout::duplicate(std::unique_ptr<CG_NormalizedGradientLayoutBase>&& dup) const
{
   dup.reset(new NormalizedGradientLayout(*this));
}

