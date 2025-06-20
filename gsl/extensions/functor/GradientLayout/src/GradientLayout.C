#include "Mgs.h"
#include "GradientLayout.h"
#include "CG_GradientLayoutBase.h"
#include "GslContext.h"
#include "ShallowArray.h"
#include "VolumeOdometer.h"
#include "Grid.h"
#include "LayerDefinitionContext.h"
#include <vector>
#include <map>
#include <memory>
#include <cmath>

void GradientLayout::userInitialize(GslContext* CG_c, int& total, double& slope, ShallowArray< int >& origin, int& originDensity, ShallowArray< bool >& gradientDimensions) 
{
}

ShallowArray< int > GradientLayout::userExecute(GslContext* CG_c) 
{
  ShallowArray<int> rval;
  Grid* g = CG_c->layerContext->grid;
  int nbrGridPts = g->getNbrGridNodes();
  for (int i=0; i<nbrGridPts; ++i) rval.push_back(0);

  std::vector<int> gridSize = g->getSize();
  unsigned dims = g->getDimensions();

  std::map<double, std::vector<int> > distanceIndexMap;
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
    distanceIndexMap[sqrt(sumsqs)].push_back(g->getNodeIndex(next));
    vo.next();
  }
  int remaining=init.total;
  std::map<double, std::vector<int> >::iterator miter=distanceIndexMap.begin(),
    mend=distanceIndexMap.end();
  for (; miter!=mend && remaining>0; ++miter) {
    int density=init.originDensity+int(init.slope*miter->first);
    density = (density<0) ? 0 : density;
    std::vector<int>::iterator viter=miter->second.begin(),
      vend=miter->second.end();
    for (; viter!=vend && remaining>0; ++viter) {
      density = (remaining-density<0) ? remaining : density;
      rval[*viter]=density;
      remaining-=density;
    }
  }
  if (remaining>0)
    std::cerr<<"WARNING: GradientLayout functor has "<<remaining
	     <<" nodes left over. Adjust slope."<<std::endl; 
  return rval;
}

GradientLayout::GradientLayout() 
   : CG_GradientLayoutBase()
{
}

GradientLayout::~GradientLayout() 
{
}

void GradientLayout::duplicate(std::unique_ptr<GradientLayout>&& dup) const
{
   dup.reset(new GradientLayout(*this));
}

void GradientLayout::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new GradientLayout(*this));
}

void GradientLayout::duplicate(std::unique_ptr<CG_GradientLayoutBase>&& dup) const
{
   dup.reset(new GradientLayout(*this));
}

