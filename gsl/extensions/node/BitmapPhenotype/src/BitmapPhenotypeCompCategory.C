// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Mgs.h"
#include "BitmapPhenotypeCompCategory.h"
#include "NDPairList.h"
#include "CG_BitmapPhenotypeCompCategory.h"
#include "BitMapHeader.h"

#include <fstream>
#include <vector>
#include <string>

#define SHD getSharedMembers()
#define ITER getSimulation().getIteration()
#define RANK getSimulation().getRank()

BitmapPhenotypeCompCategory::BitmapPhenotypeCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList) 
   : CG_BitmapPhenotypeCompCategory(sim, modelName, ndpList), _images(0)
{
}

/***** images[GLD][IMAGE_NBR][PIXEL] *****/

BitmapPhenotypeCompCategory::~BitmapPhenotypeCompCategory()
{
  if (SHD.imagesInitialized) {
    for (int i=0; i<_gridLayerDataArraySize; ++i) {
      for (int j=0; j<SHD.nbrImages; ++j) {
	delete [] _images[i][j];
      }
      delete [] _images[i];
    }
    delete [] _images;
    
    delete [] SHD.imgRows;    
    delete [] SHD.imgCols;
  }
  SHD.imageFiles.clear();
}

void BitmapPhenotypeCompCategory::initializeShared(RNG& rng) 
{
  SHD.imagesInitialized = true;
  assert(int(SHD.imageFiles.size())==_gridLayerDataArraySize);
  std::vector<std::string>* filenames = new std::vector<std::string>[_gridLayerDataArraySize];
  for (int i=0; i<_gridLayerDataArraySize; ++i) {
    Grid* g = _gridLayerDataList[i]->getGridLayerDescriptor()->getGrid();
    std::vector<int> sz = g->getSize();
    if (sz.size()!=2) {
      std::cerr<<"BitMapPhenotype Grids must have 2 dimensions : "<<sz.size()<<" found in grid "<<g<<"!"<<std::endl;
      exit(-1);
    }
    if (SHD.gridMaxL<sz[0]) SHD.gridMaxL=sz[0];
    if (SHD.gridMaxW<sz[1]) SHD.gridMaxW=sz[1];

    assert (SHD.imageFiles[i]!="");
    std::ifstream infile(SHD.imageFiles[i].c_str());
    assert (infile.good());
    if (RANK==0) std::cerr<<"Reading "<<SHD.imageFiles[i]<<std::endl;
    std::string fname;
    infile>>fname;
    while (!infile.eof()) {
      filenames[i].push_back(fname);
      infile>>fname;
    }
    infile.close();
    if (i>0) assert(SHD.nbrImages==int(filenames[i].size()));
    else SHD.nbrImages=filenames[i].size();
  }

  SHD.imgRows = new int[SHD.nbrImages];
  SHD.imgCols = new int[SHD.nbrImages];
  
  _images=new char**[_gridLayerDataArraySize];
  for (int i=0; i<_gridLayerDataArraySize; ++i)
    _images[i] = new char*[SHD.nbrImages];

  for (int i=0; i<_gridLayerDataArraySize; ++i) {
    for (int j=0; j<SHD.nbrImages; ++j) {
      FILE* inFile = fopen(filenames[i][j].c_str(),"rb");
      if (inFile==0) {
	std::cerr<<"Bad filename: "<<filenames[i][j]<<std::endl;
	exit(-1);
      }
      std::unique_ptr<char> image_aptr;

      BitMapHeader bmp;
      bmp.readGrayscales(inFile, image_aptr);
      if (i>0) {
	assert(SHD.imgRows[j]==int(bmp.infoHeader.height));
	assert(SHD.imgCols[j]==int(bmp.infoHeader.width));
      }
      else {
	SHD.imgRows[j]=bmp.infoHeader.height;
	SHD.imgCols[j]=bmp.infoHeader.width;
      }
      _images[i][j]=image_aptr.release();
      if (RANK==0)
	std::cerr<<filenames[i][j]<<"["<<SHD.imgRows[j]
		 <<","<<SHD.imgCols[j]<<"] "<<std::endl;
      assert (SHD.imgRows[j]>=SHD.gridMaxL);
      assert (SHD.imgCols[j]>=SHD.gridMaxW);
      fclose(inFile);
    }
  }
  if (RANK==0) std::cerr<<std::endl;

  ShallowArray<BitmapPhenotype>::iterator nodesIter=_nodes.begin(), nodesEnd=_nodes.end();
  for (; nodesIter!=nodesEnd; ++nodesIter) {
    int gridLayer = nodesIter->getGridLayerData()->getGridLayerIndex();
    nodesIter->image=_images[gridLayer];
  }
  updateShared(rng);
  delete [] filenames;

  int imgNbr = SHD.imageNbr = irandom(0,SHD.nbrImages-1,rng);
  SHD.row=irandom(0,SHD.imgRows[imgNbr]-SHD.gridMaxL,rng);
  SHD.col=irandom(0,SHD.imgCols[imgNbr]-SHD.gridMaxW,rng);
}

void BitmapPhenotypeCompCategory::updateShared(RNG& rng) 
{
  int phase = ITER % SHD.period;
  if(phase==0) {
    int imgNbr = SHD.imageNbr = irandom(0,SHD.nbrImages-1,rng);
    SHD.row=irandom(0,SHD.imgRows[imgNbr]-SHD.gridMaxL,rng);
    SHD.col=irandom(0,SHD.imgCols[imgNbr]-SHD.gridMaxW,rng);
  }
}
