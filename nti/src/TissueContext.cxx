// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "TissueContext.h"
#include "VecPrim.h"
#ifdef USING_BLUEGENE
#endif
#include <algorithm>

TissueContext::TissueContext()
  : _nCapsules(0), _capsules(0), _origin(0), _neuronPartitioner(0),
    _decomposition(0), _tissue(0), _boundarySynapseGeneratorSeed(0),
    _localSynapseGeneratorSeed(0), _initialized(false),
    _seeded(false), _rank(0), _mpiSize(0)
{
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_mpiSize);
}

TissueContext::~TissueContext()
{
  if (_decomposition!=_neuronPartitioner) delete _decomposition;
  delete _neuronPartitioner;
  delete [] _origin;
  delete _tissue;
}

void TissueContext::readFromFile(FILE* data, int size, int rank)
{
  PosType dataPos;
#ifdef BINARY64BITS
  fseeko64(data, (rank-size)*sizeof(PosType), SEEK_END);
  fread(&dataPos, sizeof(PosType), 1, data);
  fseeko64(data, dataPos, SEEK_SET);
#else
  fseek(data, (rank-size)*sizeof(PosType), SEEK_END);
  size_t s=fread(&dataPos, sizeof(PosType), 1, data);
  fseek(data, dataPos, SEEK_SET);
#endif
  s=fread(&_boundarySynapseGeneratorSeed, sizeof(long), 1, data);
  s=fread(&_localSynapseGeneratorSeed, sizeof(long), 1, data);
  s=fread(&_nCapsules, sizeof(int), 1, data);
  if (_nCapsules>0) {
    _capsules = new Capsule[_nCapsules];
    int offset;
    s=fread(&offset, sizeof(int), 1, data);
    _origin=_capsules+offset;
    for (int sid=0;sid<_nCapsules; ++sid) {
      _capsules[sid].readFromFile(data);
      int pass;
      s=fread(&pass, sizeof(int), 1, data);
      addToCapsuleMap(_capsules[sid].getKey(), sid, DetectionPass(pass));
    }
    int length;
    s=fread(&length, sizeof(int), 1, data);
    _touchVector.clear();
    for (int i=0; i<length; ++i) {
      Touch t;
      t.readFromFile(data);
      _touchVector.push_back(t, 0);
    }
  }
}

void TissueContext::writeToFile(int size, int rank)
{
  PosType dataPos=0, *dataPositions=0;
  if (rank==0) {
    dataPositions = new PosType[size];
    FILE* data = fopen(_commandLine.getBinaryFileName().c_str(), "wb");
    if (data == NULL)
      std::cerr<<"Warning: binary file "<<_commandLine.getBinaryFileName()<<" could not be written!"<<std::endl<<std::endl;
    else {
      _decomposition->writeToFile(data);
      fclose(data);
    }
  }
  int written=0, nextToWrite=0;
  while (nextToWrite<size) {
    if (nextToWrite==rank) {
      FILE* data = fopen(_commandLine.getBinaryFileName().c_str(), "ab");
      if (data == NULL)
	std::cerr<<"Warning: binary file "<<_commandLine.getBinaryFileName()<<" could not be written!"<<std::endl<<std::endl;
      else {
#ifdef BINARY64BITS
	dataPos=ftello64(data);
#else
	dataPos=ftell(data);
#endif
	writeData(data);
	fclose(data);
      }
      written=1;
    }
    MPI_Allreduce((void*)&written, (void*)&nextToWrite, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }
  MPI_Gather(&dataPos, 1, MPI_POS_TYPE, dataPositions, 1, MPI_POS_TYPE, 0, MPI_COMM_WORLD);
  if (rank==0) {
    FILE* data = fopen(_commandLine.getBinaryFileName().c_str(), "ab");
    if (data == NULL)
      std::cerr<<"Warning: binary file "<<_commandLine.getBinaryFileName()<<" could not be written!"<<std::endl<<std::endl;
    else {
      fwrite(dataPositions, sizeof(PosType), size, data);
      fclose(data);
    }
  }
}

void TissueContext::writeData(FILE* data)
{
  fwrite(&_boundarySynapseGeneratorSeed, sizeof(long), 1, data);
  fwrite(&_localSynapseGeneratorSeed, sizeof(long), 1, data);
  fwrite(&_nCapsules, sizeof(int), 1, data);
  if (_nCapsules>0) {
    int offset=_origin-_capsules;
    fwrite(&offset, sizeof(int), 1, data);
    for (int sid=0;sid<_nCapsules; ++sid) {
      _capsules[sid].writeToFile(data);
      int pass = (int)(getPass(_capsules[sid].getKey()));
      fwrite(&pass, sizeof(int), 1, data);
    }
    int length=_touchVector.getCount();
    fwrite(&length, sizeof(int), 1, data);
    TouchVector::TouchIterator titer=_touchVector.begin(), 
      tend=_touchVector.end();
    for (; titer!=tend; ++titer) {
      titer->writeToFile(data);
    }
  }
}

int TissueContext::setUpCapsules(int nCapsules, DetectionPass detectionPass,int rank, int maxComputeOrder)
{
  Capsule* capsEnd=_capsules+nCapsules;
  resetBranches();
  std::sort(_capsules, capsEnd);
  capsEnd=std::unique(_capsules, capsEnd);
  _nCapsules = capsEnd - _capsules;
  if (detectionPass != NOT_SET) {
    for(int sid=0; sid<_nCapsules; ++sid) {
      addToCapsuleMap(_capsules[sid].getKey(), sid, detectionPass);
    }
  }
  setUpBranches(rank, maxComputeOrder);
  return _nCapsules;
}

void TissueContext::setUpBranches(int rank, int maxComputeOrder)
{
  int i=0;
  while (i<_nCapsules) {
    ComputeBranch* branch=new ComputeBranch; 
    branch->_capsules=&_capsules[i];
    branch->_nCapsules=1;
    double key=_capsules[i].getKey();
    unsigned int neuronIndex=_segmentDescriptor.getNeuronIndex(key);
    unsigned int branchIndex=_segmentDescriptor.getBranchIndex(key); 
    unsigned int computeOrder=_segmentDescriptor.getComputeOrder(key);
    DetectionPass branchPass=getPass(key);

    if (!isSpanning(_capsules[i]) && branchPass==FIRST_PASS ) {
      int j=i;
      while (++j<_nCapsules && sameBranch(_capsules[j], neuronIndex, branchIndex, computeOrder, branchPass) && isConsecutiveCapsule(j) ) {
	branch->_nCapsules++;
	// Remember: computeOrder is not sufficient to detect slice boundary branch termination in the case MaxComputeOrder==0
	if (isSpanning(_capsules[j]) ) break;
      }
    }
    i+=branch->_nCapsules;

    std::map<unsigned int, std::vector<ComputeBranch*> >::iterator
      mapIter=_neurons.find(neuronIndex);
    double* coords=branch->_capsules[0].getBeginCoordinates();
    unsigned int branchOrder=_segmentDescriptor.getBranchOrder(key);
    if (mapIter!=_neurons.end()) {
      if (branchPass==FIRST_PASS) {
	std::vector<ComputeBranch*>& branches=mapIter->second;
	int nBranches=branches.size();
	for (int j=0; j<nBranches; ++j) {
	  ComputeBranch* candidateBranch=branches[j];
	  if (getPass(candidateBranch->_capsules[0].getKey())==FIRST_PASS &&
	      !isOutside(candidateBranch, rank)) {
	    unsigned int candidateBranchIndex=
	      _segmentDescriptor.getBranchIndex(candidateBranch->_capsules[0].getKey());
	    unsigned int candidateOrder=
	      _segmentDescriptor.getBranchOrder(candidateBranch->_capsules[0].getKey());
	    unsigned int candidateComputeOrder=
	      _segmentDescriptor.getComputeOrder(candidateBranch->lastCapsule().getKey());
	    double* candidateCoords=candidateBranch->lastCapsule().getEndCoordinates();
	    
	    if ( (candidateOrder==branchOrder-1 || candidateBranchIndex==branchIndex) &&
		 SqDist(coords, candidateCoords)==0.0 ) {
	      branch->_parent=candidateBranch;
	      branch->_parent->_daughters.push_back(branch);
	      if (computeOrder==0 && candidateComputeOrder!=maxComputeOrder) {
		std::cerr<<"TissueContext("<<rank<<") : Mismatched compute orders (1,"
			 <<candidateComputeOrder<<"!="<<maxComputeOrder<<")!"<<std::endl;
		assert(0);
	      }
	      if (computeOrder>0 && candidateComputeOrder!=computeOrder-1) {
		std::cerr<<"TissueContext("<<rank<<") : Mismatched compute orders (2,"
			 <<candidateComputeOrder<<"!="<<computeOrder-1<<")!"<<std::endl;
		assert(0);
	      }
	      break;
	    }
	  }
	}
      }
    }
    else {
      std::vector<ComputeBranch*> newBranchVector;
      _neurons[neuronIndex]=newBranchVector;
    }
    for (int j=0; j<branch->_nCapsules; ++j) {
      branch->_capsules[j].setBranch(branch);
    }
    _neurons[neuronIndex].push_back(branch);
  }
}

void TissueContext::resetBranches()
{
  std::map<unsigned int, std::vector<ComputeBranch*> >::iterator mapIter, mapEnd=_neurons.end();
  for (mapIter=_neurons.begin(); mapIter!=mapEnd; ++mapIter) {
    std::vector<ComputeBranch*>& branches=mapIter->second;
    for (int i=0; i<branches.size(); ++i) {
      delete branches[i];
    }
    branches.clear();
  }
  _neurons.clear();
}

bool TissueContext::sameBranch(Capsule& capsule, 
			       unsigned int neuronIndex, 
			       unsigned int branchIndex, 
			       unsigned int computeOrder,
			       DetectionPass branchPass)
{
  return ( _segmentDescriptor.getNeuronIndex(capsule.getKey()) == neuronIndex &&
	   _segmentDescriptor.getBranchIndex(capsule.getKey()) == branchIndex && 
	   _segmentDescriptor.getComputeOrder(capsule.getKey()) == computeOrder &&
	   getPass(capsule.getKey()) == branchPass );
}

bool TissueContext::isGoing(Capsule& capsule, int rank)
{
  assert(_decomposition && rank>=0);
  Sphere endSphere;
  capsule.getEndSphere(endSphere);
  return (_decomposition->getRank(capsule.getSphere())==rank &&
	  _decomposition->getRank(endSphere)!=rank);
}

bool TissueContext::isComing(Capsule& capsule, int rank)
{
  assert(_decomposition && rank>=0);
  Sphere endSphere;
  capsule.getEndSphere(endSphere);
  return (_decomposition->getRank(capsule.getSphere())!=rank &&
	  _decomposition->getRank(endSphere)==rank);
}

bool TissueContext::isSpanning(Capsule& capsule)
{
  Sphere endSphere;
  capsule.getEndSphere(endSphere);
  return (_decomposition->getRank(capsule.getSphere())!=
	  _decomposition->getRank(endSphere));  
}

bool TissueContext::isOutside(ComputeBranch* branch, int rank)
{
  assert(_decomposition && rank>=0);
  Sphere endSphere;
  branch->lastCapsule().getEndSphere(endSphere);
  return (_decomposition->getRank(branch->_capsules->getSphere())!=rank &&
	  _decomposition->getRank(endSphere)!=rank);
}

bool TissueContext::isConsecutiveCapsule(int index)
{
  assert(index>0 && index<_nCapsules);
  return (_segmentDescriptor.getSegmentIndex(_capsules[index].getKey())-_segmentDescriptor.getSegmentIndex(_capsules[index-1].getKey())==1);
}

unsigned int TissueContext::getRankOfBeginPoint(ComputeBranch* branch)
{
  return _decomposition->getRank(branch->_capsules[0].getSphere());
}

unsigned int TissueContext::getRankOfEndPoint(ComputeBranch* branch)
{
  Sphere endSphere;
  branch->lastCapsule().getEndSphere(endSphere);
  return _decomposition->getRank(endSphere);
}

bool TissueContext::isTouchToEnd(Capsule& c, Touch& t)
{
  bool rval=false;
  rval=(c.getEndProp()<=t.getProp(c.getKey()));
  return rval;
}

bool TissueContext::isMappedTouch(Touch& t, std::map<double, int>::iterator& iter1, std::map<double, int>::iterator& iter2)
{
  double s1Key=t.getKey1();
  bool rval=( (iter1=_firstPassCapsuleMap.find(s1Key) ) != _firstPassCapsuleMap.end() );
  if (!rval) rval=( (iter1=_secondPassCapsuleMap.find(s1Key)) != _secondPassCapsuleMap.end() );
  if (rval) {
    double s2Key=t.getKey2();
    rval=( (iter2=_firstPassCapsuleMap.find(s2Key) ) != _firstPassCapsuleMap.end() );
    if (!rval) rval=( (iter2=_secondPassCapsuleMap.find(s2Key) ) != _secondPassCapsuleMap.end() );
  }
  return rval;
}

bool TissueContext::isLensTouch(Touch& t, int rank)
{
  bool rval=false;
  std::map<double, int>::iterator iter1, iter2;
  if (isMappedTouch(t, iter1, iter2)) {
    double s1Key=t.getKey1();
    Capsule& c1=_capsules[iter1->second];
    bool preJct=(_segmentDescriptor.getFlag(s1Key) && isTouchToEnd(c1,t) );
    Sphere endSphere1;
    c1.getEndSphere(endSphere1);
    rval=( preJct ? rank==_decomposition->getRank(endSphere1) : rank==_decomposition->getRank(c1.getSphere()) );
    if (!rval) {
      double s2Key=t.getKey2();
      Capsule& c2=_capsules[iter2->second];
      bool postJct=(_segmentDescriptor.getFlag(s2Key) && isTouchToEnd(c2,t) );		      
      Sphere endSphere2;
      c2.getEndSphere(endSphere2);
      rval=( postJct ? rank==_decomposition->getRank(endSphere2) : rank==_decomposition->getRank(c2.getSphere()) );
    }
  }
  return rval;
}

void TissueContext::correctTouchKeys(int rank)
{
  SegmentDescriptor segmentDescriptor;
  TouchVector::TouchIterator tend=_touchVector.end();
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  maskVector.push_back(SegmentDescriptor::neuronIndex);
  maskVector.push_back(SegmentDescriptor::branchIndex);
  maskVector.push_back(SegmentDescriptor::segmentIndex);
  unsigned long long mask = _segmentDescriptor.getMask(maskVector);

  for (TouchVector::TouchIterator titer=_touchVector.begin(); titer!=tend; ++titer) {
    double key1 = _segmentDescriptor.getSegmentKey(titer->getKey1(), mask);
    double key2 = _segmentDescriptor.getSegmentKey(titer->getKey2(), mask);
    std::map<double, int>::iterator mapiter, mapend=_firstPassCapsuleMap.end();
    bool key1Fix=false, key2Fix=false;
    for (mapiter = _firstPassCapsuleMap.begin(); mapiter!=mapend && !(key1Fix && key2Fix); ++mapiter) {
      double capkey = _segmentDescriptor.getSegmentKey(mapiter->first, mask);
      if (!key1Fix) {if (key1Fix=(capkey==key1)) titer->setKey1(mapiter->first);}
      if (!key2Fix) {if (key2Fix=(capkey==key2)) titer->setKey2(mapiter->first);}
    }
    mapend=_secondPassCapsuleMap.end();
    for (mapiter = _secondPassCapsuleMap.begin(); mapiter!=mapend && !(key1Fix && key2Fix); ++mapiter) {
      double capkey = _segmentDescriptor.getSegmentKey(mapiter->first, mask);
      if (!key1Fix) {if (key1Fix=(capkey==key1)) titer->setKey1(mapiter->first);}
      if (!key2Fix) {if (key2Fix=(capkey==key2)) titer->setKey2(mapiter->first);}
    }
    assert(key1Fix&&key2Fix);
  }
}

TissueContext::DetectionPass TissueContext::addToCapsuleMap(double key, int index, DetectionPass detectionPass)
{
  DetectionPass rval=FIRST_PASS;
  if (detectionPass==FIRST_PASS) {
    _firstPassCapsuleMap[key]=index;
  }
  else {
    std::map<double, int>::iterator mapiter = _firstPassCapsuleMap.find(key);
    if (mapiter==_firstPassCapsuleMap.end()) {
      _secondPassCapsuleMap[key]=index;
      rval=SECOND_PASS;
    }
    else (*mapiter).second=index;
  }
  return rval;
}

int TissueContext::getCapsuleIndex(double key)
{
  std::map<double, int>::iterator mapiter = _firstPassCapsuleMap.find(key);
  if (mapiter==_firstPassCapsuleMap.end()) {
    mapiter=_secondPassCapsuleMap.find(key);
    assert (mapiter!=_secondPassCapsuleMap.end());
  }
  return (*mapiter).second;
}

TissueContext::DetectionPass TissueContext::getPass(double key)
{
  DetectionPass rval=FIRST_PASS;
  if (_firstPassCapsuleMap.find(key)==_firstPassCapsuleMap.end()) {
    rval=SECOND_PASS;
    assert (_secondPassCapsuleMap.find(key)!=_secondPassCapsuleMap.end());
  }
  return rval;
}

void TissueContext::clearCapsuleMaps()
{
  _firstPassCapsuleMap.clear();
  _secondPassCapsuleMap.clear();
}

void TissueContext::seed(int rank)
{
  if (!_seeded) {
    RNG rangen, sharedRangen;
    rangen.reSeed(_commandLine.getSeed(), rank);
    sharedRangen.reSeedShared( _commandLine.getSeed()-1 );
    _boundarySynapseGeneratorSeed=lrandom(sharedRangen);
    _localSynapseGeneratorSeed=lrandom(rangen);
    _seeded=true;
  }
  _localSynapseGenerator.reSeed(_localSynapseGeneratorSeed, rank );
  _touchSampler.reSeed(lrandom(_localSynapseGenerator), rank);
}

void TissueContext::getCapsuleMaps(std::map<double, int>& firstPassCapsuleMap, std::map<double,int>& secondPassCapsuleMap)
{
  firstPassCapsuleMap=_firstPassCapsuleMap;
  secondPassCapsuleMap=_secondPassCapsuleMap;  
}

void TissueContext::resetCapsuleMaps(std::map<double, int>& firstPassCapsuleMap, std::map<double,int>& secondPassCapsuleMap)
{
  clearCapsuleMaps();
  _firstPassCapsuleMap=firstPassCapsuleMap;
  _secondPassCapsuleMap=secondPassCapsuleMap;
}

void TissueContext::rebalance(Params* params, TouchVector* touchVector)
{
  TouchVector::TouchIterator tend=touchVector->end();

  int ** localHistogram;
  double *minXYZ, *maxXYZ, *binwidth; 
  int *nbinsXYZ;
  _tissue->getLocalHistogram(localHistogram, minXYZ, maxXYZ, binwidth, nbinsXYZ);
  double** costHistogram=new double*[3];
  for (int d=0; d<3; ++d) {
    costHistogram[d]=new double[nbinsXYZ[d]];
    for (int i=0; i < nbinsXYZ[d]; ++i) {
      costHistogram[d][i]=0;
    }
  }

  for (int i=0; i<_nCapsules; ++i) {
    if (_decomposition->getRank(_capsules[i].getSphere())==_rank) {	
      double* coords=_capsules[i].getBeginCoordinates();
      double key = _capsules[i].getKey();
      
      std::list<std::string> const * compartmentVariableTargets =params->getCompartmentVariableTargets(key);
      if (compartmentVariableTargets) {
	std::list<std::string>::const_iterator iter=compartmentVariableTargets->begin(), end=compartmentVariableTargets->end();
	for (; iter!=end; ++iter) {
	  double cost=params->getCompartmentVariableCost(*iter);
	  for (int d=0; d<3; ++d) {
	    int bin = int((coords[d]-minXYZ[d])/binwidth[d]);
	    if (bin>=nbinsXYZ[d]) bin=nbinsXYZ[d]-1;
	    costHistogram[d][bin]+=cost;
	  }
	}
      }

      std::list<Params::ChannelTarget> const * channelTargets =params->getChannelTargets(key);
      if (channelTargets) {
	std::list<Params::ChannelTarget>::const_iterator iter=channelTargets->begin(), end=channelTargets->end();
	for (; iter!=end; ++iter) {
	  double cost=params->getChannelCost((*iter)._type);
	  for (int d=0; d<3; ++d) {
	    int bin = int((coords[d]-minXYZ[d])/binwidth[d]);
	    if (bin>=nbinsXYZ[d]) bin=nbinsXYZ[d]-1;
	    costHistogram[d][bin]+=cost;
	  }
	}
      }
    }
  } 
  for (int direction=0; direction<2; ++direction) {
    for (TouchVector::TouchIterator titer=touchVector->begin(); titer!=tend; ++titer) { 
      double key1, key2;
      if (direction==0) {
	key1 = titer->getKey1();
	key2 = titer->getKey2();
      }
      else {
	key1 = titer->getKey2();
	key2 = titer->getKey1();
      }
      int c1Idx=getCapsuleIndex(key1);
      int c2Idx=getCapsuleIndex(key2);
      Capsule& c1=_capsules[c1Idx];
      Capsule& c2=_capsules[c2Idx];
      double* coords1=c1.getBeginCoordinates();
      double* coords2=c2.getBeginCoordinates();
      if (params->electricalSynapses()  && (key1<key2 || !params->symmetricElectricalSynapseTargets(key1, key2) ) ) {
	std::list<Params::ElectricalSynapseTarget> const * electricalSynapseTargets=
	  params->getElectricalSynapseTargets(key1, key2);
	if (electricalSynapseTargets) {
	  std::list<Params::ElectricalSynapseTarget>::const_iterator iiter=electricalSynapseTargets->begin(),
	    iend=electricalSynapseTargets->end();
	  for (; iiter!=iend; ++iiter) {
	    double cost=params->getElectricalSynapseCost((*iiter)._type)*(*iiter)._parameter;
	    for (int d=0; d<3; ++d) {
	      int bin1 = int((coords1[d]-minXYZ[d])/binwidth[d]);
	      if (bin1>=nbinsXYZ[d]) bin1=nbinsXYZ[d]-1;
	      int bin2 = int((coords2[d]-minXYZ[d])/binwidth[d]);
	      if (bin2>=nbinsXYZ[d]) bin2=nbinsXYZ[d]-1;
	      costHistogram[d][bin1]+=cost;
	      costHistogram[d][bin2]+=cost;		
	    }
	  }
	}
      }
      if (params->chemicalSynapses()) {
	std::list<Params::ChemicalSynapseTarget> const * chemicalSynapseTargets=
	  params->getChemicalSynapseTargets(key1, key2);
	if (chemicalSynapseTargets) {
	  std::list<Params::ChemicalSynapseTarget>::const_iterator iiter=chemicalSynapseTargets->begin(),
	    iend=chemicalSynapseTargets->end();
	  for (; iiter!=iend; ++iiter) {
	    std::map<std::string, std::pair<std::list<std::string>, std::list<std::string> > >::const_iterator targetsIter=iiter->_targets.begin();
	    double cost=0;
	    for (; targetsIter!=iiter->_targets.end(); ++targetsIter) {
	      cost+=params->getChemicalSynapseCost(targetsIter->first)*(*iiter)._parameter;
	    }
	    for (int d=0; d<3; ++d) {
	      int bin = int((coords1[d]-minXYZ[d])/binwidth[d]);
	      if (bin>=nbinsXYZ[d]) bin=nbinsXYZ[d]-1;
	      costHistogram[d][bin]+=cost;
	    }
	  }
	}
      }
    }
  }
 
  for (int d=0; d<3; ++d) {
    for (int i=0; i < nbinsXYZ[d]; ++i) {
      int n=localHistogram[d][i]=int(SIG_HIST*costHistogram[d][i]);
    }
    delete [] costHistogram[d];
  }
  delete [] costHistogram;
  
  _tissue->generateAlternateHistogram();
  _decomposition->decompose();

  for (TouchVector::TouchIterator titer=touchVector->begin(); titer!=tend; ++titer) {
    double key1 = titer->getKey1();
    double key2 = titer->getKey2();
    int c1Idx=getCapsuleIndex(key1);
    int c2Idx=getCapsuleIndex(key2);
    Capsule& c1=_capsules[c1Idx];
    Capsule& c2=_capsules[c2Idx];

    ShallowArray<int, MAXRETURNRANKS, 100> ranks1, ranks2, ranks;

    _decomposition->getRanks(&c1.getSphere(), c1.getEndCoordinates(), params->getRadius(key1), ranks1);
    _decomposition->getRanks(&c2.getSphere(), c2.getEndCoordinates(), params->getRadius(key2), ranks2);    
    ranks=ranks1;
    ranks.merge(ranks2);

    int capRank1=_decomposition->getRank(c1.getSphere());
    int capRank2=_decomposition->getRank(c2.getSphere());

    ShallowArray<int, MAXRETURNRANKS, 100>::iterator ranksIter=ranks.begin(), ranksEnd=ranks.end();
    if (ranksIter!=ranksEnd) {
      int idx=*ranksIter;
      ++ranksIter;
      for (; ranksIter!=ranksEnd; ++ranksIter) {
	if (idx==*ranksIter) {
	  touchVector->mapTouch(idx, titer);
	  break;
	}
	idx=*ranksIter;
      }
    }
  }
}
