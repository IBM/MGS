// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "Branch.h"
#include "Neurogenesis.h"
#include "Segment.h"
#include "Neuron.h"
#include "VecPrim.h"
#include "tangent.h"
#include <list>
#include <algorithm>
#include <float.h>

#define DEFAULT_RADIUS 0.001

Branch::Branch()
  : _branchType(0), _branchOrder(-1), _dist2Soma(0), _numberOfSegments(0), _branchIndex(0), 
    _segments(0), _neuron(0), _rootSegment(0), _resampledTerminalIndex(-1)
{
}

Branch::Branch(Branch const & b)
  : _branchType(b._branchType), 
    _branchOrder(b._branchOrder), 
    _dist2Soma(b._dist2Soma),
    _numberOfSegments(b._numberOfSegments),
    _branchIndex(b._branchIndex), 
    _segments(b._segments), 
    _neuron(b._neuron), 
    _rootSegment(b._rootSegment),
    _resampledTerminalIndex(b._resampledTerminalIndex)
{
	//std::memcpy(_displacedTerminalCoords, b._displacedTerminalCoords, 3*sizeof(double));
	std::copy(b._displacedTerminalCoords, b._displacedTerminalCoords+3, _displacedTerminalCoords);
}

Branch& Branch::operator=(const Branch& b)
{
  if (this==&b) return *this;
  _branchType=b._branchType;
  _branchOrder=b._branchOrder;
  _dist2Soma=b._dist2Soma;
  _numberOfSegments=b._numberOfSegments;
  _branchIndex=b._branchIndex;
  _segments=b._segments;
  _neuron=b._neuron;
  _rootSegment=b._rootSegment;
  _resampledTerminalIndex=b._resampledTerminalIndex;
  //std::memcpy(_displacedTerminalCoords, b._displacedTerminalCoords, 3*sizeof(double));
  std::copy(b._displacedTerminalCoords, b._displacedTerminalCoords+3, _displacedTerminalCoords);
  return *this;
}

//   Revised this: should return the branch-lenght
//   instead of the distance between the first and last segment
const double Branch::getLength()
{
  assert(_numberOfSegments>0);
  //return sqrt(SqDist(_segments[0].getCoords(), _segments[_numberOfSegments-1].getCoords()));
  return _segments[_numberOfSegments-1].getDist2Soma() - _segments[0].getDist2Soma();

}

void Branch::resample(std::vector<Segment>& segments, double pointSpacing)
{
  int i=0;
  Segment newSeg;
  newSeg=_segments[0];
  int segCount=0; // track # segments in the resampled branch
  Segment *seg1, *seg2;
  double L, d=0;
  unsigned s=segments.size();
  assert(_numberOfSegments>0);
  int numberOfResampledSegments=0;

  if (_branchIndex==0) {
    if (_branchType!=0) {
      std::cerr<<"Branch.cxx : First branch must be cell body!"<<std::endl;
      exit(0);
    }
    segments.resize(s+2);
    segments[s]=_segments[0];
    segments[s].setSegmentIndex(segCount);
    segments[s].setSegmentArrayIndex(s);
    segments[s].setKey();
    ++s;
    ++segCount;
    segments[s]=_segments[0];
    segments[s].setSegmentIndex(segCount);
    segments[s].setSegmentArrayIndex(s);
    segments[s].setKey();
    ++s;
    numberOfResampledSegments=2;
    _resampledTerminalIndex=segments.size()-1;
  }
  else {
    assert(_rootSegment);
    segments.resize(s+1);
    int terminalIndex=_rootSegment->getBranch()->getResampledTerminalIndex();
    assert (terminalIndex>=0);
    segments[s] = segments[terminalIndex];
    segments[s].resetBranch(this);
    segments[s].setSegmentIndex(segCount);
    segments[s].setSegmentArrayIndex(s);
    segments[s].setKey();
    segCount=1;
    if (segments[terminalIndex].getBranch()->getBranchType()==Branch::_SOMA)
      segments[s].getRadius()=_segments[0].getRadius();
    seg1=&segments[s];
    seg2=&_segments[0];
    ++s;
   
    if (nextSegment(seg1, seg2, i, L)) {
      while(1) {
	while(1) {
	  double d=2.0*L*seg1->getRadius()/( (L/pointSpacing) + seg1->getRadius() - seg2->getRadius()); //see P1010005.jpg
	  if (fabs(d)>L || d<0) break;
	  resetCoordinates(newSeg, seg1, seg2, d, L, pointSpacing);
	  newSeg.setSegmentIndex(segCount);
	  newSeg.setSegmentArrayIndex(s);
	  newSeg.setKey();
	  segments.resize(s+1);
	  segments[s]=newSeg;
	  seg1=&segments[s];
	  ++segCount;
	  ++s;
	  L=sqrt(SqDist(seg1->getCoords(),seg2->getCoords()));
	  if (L==0) break;
	}
	if (++i==_numberOfSegments) break;
	seg2=&_segments[i];
	d=0;
	if (!nextSegment(seg1, seg2, i, L)) break;
      }
    }
    if (segCount==1 && pointSpacing>1.0) {
      double d=2.0*L*seg1->getRadius()/( L + seg1->getRadius() - seg2->getRadius()); //see P1010005.jpg
      if (fabs(d)<=L) {
	newSeg=_segments[0];
	newSeg.setSegmentIndex(segCount);
	newSeg.setSegmentArrayIndex(s);
	newSeg.setKey();
	segments.resize(s+1);
	segments[s]=newSeg;
	++segCount;
	++s;
      }
    }
    if (segCount>1) {
      numberOfResampledSegments=segCount; 
      _resampledTerminalIndex=segments.size()-1;
    }
    else {
      numberOfResampledSegments=0;
      _resampledTerminalIndex=_rootSegment->getBranch()->getResampledTerminalIndex();
      segments.resize(--s);
    }
  }
  _numberOfSegments=numberOfResampledSegments;
}

void Branch::resetBranchRoots(std::vector<Segment>& segments)
{
  int segCount=0;
  unsigned s=segments.size();
  assert(_numberOfSegments>0);

  if (_branchIndex==0) { 
    if (_branchType!=0) {
      std::cerr<<"Branch.cxx : First branch must be cell body!"<<std::endl;
      exit(0);
    }
    segments.resize(s+2);
    segments[s]=_segments[0];
    segments[s].setSegmentIndex(segCount);
    segments[s].setSegmentArrayIndex(s);
    segments[s].setKey();
    ++segCount;
    ++s;
    segments[s]=_segments[0];
    segments[s].setSegmentIndex(segCount);
    segments[s].setSegmentArrayIndex(s);
    segments[s].setKey();
    ++segCount;
    ++s;
  }
  else {
    assert(_rootSegment);
    double* rootSegCoords=_rootSegment->getCoords();
    double* firstSegCoords=_segments[0].getCoords();
    bool newRoot=false;
    for (int k=0; k<3; ++k) newRoot=newRoot || (rootSegCoords[k]!=firstSegCoords[k]);
    if (newRoot) {
      segments.resize(s+1);
      segments[s] = *_rootSegment;
      segments[s].resetBranch(this);
      segments[s].setSegmentIndex(segCount);
      segments[s].setSegmentArrayIndex(s);
      segments[s].setKey();
      ++segCount;
      ++s;
    }
    for (int i=0; i<_numberOfSegments; ++i) {
      segments.resize(s+1);
      segments[s] = _segments[i];
      segments[s].resetBranch(this);
      segments[s].setSegmentIndex(segCount);
      segments[s].setSegmentArrayIndex(s);
      segments[s].setKey();
      ++s;
      ++segCount;
    }
    if (newRoot && _rootSegment->getBranch()->getBranchIndex()==0) {
      segments[s-segCount].
	getRadius()=segments[s-segCount+1].getRadius();
    }     
  }
  _numberOfSegments=segCount;
}

Segment* Branch::loadBinary(
		      FILE* inputDataFile,
		      Segment* segmentPtr,
		      Neuron* neuron,
		      const int branchIndex)
{

  _neuron = neuron;
  
  _branchIndex = branchIndex;
  
  size_t s=fread(&_branchType, sizeof(int), 1, inputDataFile);
  s=fread(&_numberOfSegments, sizeof(int), 1, inputDataFile);
  //printf("branchType=%d %d\n",_branchType,_numberOfSegments);

  _segments = segmentPtr;

  for(int i=0; i<_numberOfSegments; ++i, ++segmentPtr) {   
    //upon break out, segmentPtr will point to the beginning of the next position for writing segments
    segmentPtr->loadBinary(inputDataFile, this, i);
    if (segmentPtr->getRadius()==0)
      segmentPtr->getRadius()=DEFAULT_RADIUS;
  }
  setDisplacedTerminalCoords();
  return segmentPtr;
}

Segment* Branch::loadText(
		      FILE* inputDataFile,
		      Segment* segmentPtr,
		      Neuron* neuron,
		      const int branchIndex,
		      std::list<int>& branchTerminals, 
		      double xOffset, 
		      double yOffset, 
		      double zOffset,
		      int cellBodyCorrection)
{
  _neuron = neuron;
  _branchIndex = branchIndex;
  
  int seg, parent, prevSeg, nextBranchType;
  float x, y, z, r;
  int f=fscanf(inputDataFile, "%d %d %f %f %f %f %d", &seg, &_branchType, &x, &y, &z, &r, &parent);
  --_branchType; // Necessary because Neuromorpho.org orders branchtypes from 1
  if (_branchType!=0) {
    if (parent!=1) parent-=cellBodyCorrection;
    seg-=cellBodyCorrection;
    _rootSegment=&neuron->getSegmentsBegin()[parent-1];
  }
  else { // correction added when NeuroMorpho.org allowed multiple cell bodies in a single swc file
    int tmpSeg, tmpParent, tmpBranchType, pos=ftell(inputDataFile);
    float tmpx=0.0, tmpy=0.0, tmpz=0.0, tmpr=0.0;
    do {
      x+=tmpx;
      y+=tmpy;
      z+=tmpz;
      r+=tmpr;
      pos=ftell(inputDataFile);
      f=fscanf(inputDataFile, "%d %d %f %f %f %f %d", &tmpSeg, &tmpBranchType, &tmpx, &tmpy, &tmpz, &tmpr, &tmpParent);
    } while (tmpBranchType==1 && tmpParent==1);
    fseek(inputDataFile, pos, SEEK_SET);
    x/=double(cellBodyCorrection+1);
    y/=double(cellBodyCorrection+1);
    z/=double(cellBodyCorrection+1);
    r/=double(cellBodyCorrection+1);
  }
  _numberOfSegments=0;
  prevSeg=parent;
  nextBranchType=_branchType;
  _segments = segmentPtr;
  int pos2;
  while (1) {
    if (seg>1 && _branchType==0) {
      std::cerr<<"Branch.cxx : Only one cell body allowed per file!"<<std::endl;
      exit(0);
    }
    assert(parent<seg);    
    if (r==0) r=DEFAULT_RADIUS;
    segmentPtr->loadText(x+xOffset, y+yOffset, z+zOffset, r, this, _numberOfSegments);
    prevSeg=seg;
    ++_numberOfSegments;
    ++segmentPtr;
    pos2=ftell(inputDataFile);
    //upon break out, segmentPtr will point to the beginning of the next position for writing segments
    if ( fscanf(inputDataFile, "%d %d %f %f %f %f %d", &seg, &nextBranchType, &x, &y, &z, &r, &parent)==EOF ||
	 find(branchTerminals.begin(), branchTerminals.end(), prevSeg)!=branchTerminals.end()) {
      break;
    }
    seg-=cellBodyCorrection;
    if (parent!=1) parent-=cellBodyCorrection;
  }
  fseek(inputDataFile, pos2, SEEK_SET);
  setDisplacedTerminalCoords();
  return segmentPtr;
}

// CONTEXT: the current branch belong to a certain neuron '_neuron'
// GOAL: Return the segment associated with the soma of that '_neuron'
//
void Branch::findRootSegment()
{
  assert (_numberOfSegments>0);
  double D = FLT_MAX;
  double* firstcoords = _segments[0].getCoords();
  Branch* branches = _neuron->getBranches();
  // traverse all branches of that '_neuron' and find the one with the 
  //    shortest distance between the 'first' and 'last' segments of a branch
  for(int k=0; k<_branchIndex; ++k) {
    if (branches[k].getNumberOfSegments()!=0) {
      Segment& lastseg=branches[k].getSegments()[branches[k].getNumberOfSegments()-1];
      double* lastcoords = lastseg.getCoords();
      double d = SqDist(firstcoords, lastcoords);
      if(d < D) {
	_rootSegment=&lastseg;
	D = d;
      }
    }
  }
  assert(D==0.0);
}

// GOAL: Set the along-branch-distances to the soma
//       from each Segment in the current Branch 
// INPUT:
//    dist2Soma = the along-branch-distance from the first Segment of that Branch to the soma
void Branch::setDist2Soma(double dist2Soma)
{
  assert(_numberOfSegments>0);
  _segments[0].setDist2Soma(_dist2Soma=dist2Soma);
  for (int i=1; i<_numberOfSegments; ++i) {
    dist2Soma+=sqrt(SqDist(_segments[i-1].getCoords(), _segments[i].getCoords()));
    _segments[i].setDist2Soma(dist2Soma);
 }
}

void Branch::resetSegments(Segment* segments)
{
  _segments=segments;
  setDisplacedTerminalCoords();
}
 

void Branch::resetSegments(Segment* segments, Segment* rootSegment)
{
  resetSegments(segments);
  _rootSegment=rootSegment;
}

void Branch::setDisplacedTerminalCoords()
{
  double* cdsT=getTerminalSegment()->getCoords();
  double* cdsCB=_neuron->getSegmentsBegin()->getCoords();
  double* cdsOrigCB=_neuron->getSegmentsBegin()->getOrigCoords();
  double dist=SqDist(cdsOrigCB, cdsT);
  for (int i=0; i<3; ++i) 
    _displacedTerminalCoords[i]=cdsCB[i]+(cdsT[i]-cdsOrigCB[i])*dist;
}

void Branch::writeCoordinates(FILE* fp)
{
  fwrite(&_branchType, sizeof(int), 1, fp);
  fwrite(&_numberOfSegments, sizeof(int), 1, fp);
  for(int i=0; i<_numberOfSegments; ++i) _segments[i].writeCoordinates(fp);
}

void Branch::resetCoordinates(Segment& newSeg, Segment* seg2,
			      Segment* seg3, double d, double L, double pointSpacing)
{
  assert(d>0);
  double ratio = d/L;
  double r2=seg2->getRadius(), r3=seg3->getRadius();
  newSeg.getRadius()=d/pointSpacing-r2;
  double* c2=seg2->getCoords();
  double* c3=seg3->getCoords();
  double* newSegOrigCds=newSeg.getOrigCoords();
  double* newSegCds=newSeg.getCoords();
  for (int i=0; i<3; ++i) {
    newSegOrigCds[i]=newSegCds[i]=(c2[i]+(c3[i]-c2[i])*ratio);
  }
  double m = newSeg.getRadius()+MIN_MASS;
  newSeg.setMass(m*m*m);
  newSeg.setKey();
}

bool Branch::nextSegment(Segment* seg1, Segment*& seg2, int& i, double& L)
{
  bool success=true;
  L=sqrt(SqDist(seg1->getCoords(),seg2->getCoords()));
  while (L<seg1->getRadius() || L<seg2->getRadius()) {
    if (++i==_numberOfSegments) {
      success=false;
      break;
    }
    seg2=&_segments[i];
    L=sqrt(SqDist(seg1->getCoords(),seg2->getCoords()));
  }
  return success;
}

bool Branch::isTerminalBranch()
{
  return (_branchOrder==_neuron->getMaxBranchOrder());
}
