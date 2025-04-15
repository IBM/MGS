#ifndef BITMAPHEADER_H
#define BITMAPHEADER_H
// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
#include <string>
#include <inttypes.h>
#include <memory>

#define FILEHEADER_SIZE 14
#define INFOHEADER_SIZE 40
#define PALETTE_SIZE 1024

#define FILEHEADER_ID_LENGTH 2

class BitMapHeader 

{
 public:
  BitMapHeader();
  BitMapHeader(char* id, uint32_t fsz, uint16_t rsvd1, uint16_t rsvd2, uint32_t bmDataOffset, uint32_t bmHeaderSize, uint32_t w, uint32_t h, uint16_t p, uint16_t bpp, uint32_t c, uint32_t bmDataSize, uint32_t hRes, uint32_t vRes, uint32_t clrs, uint32_t impClrs);

  std::string getHeaderInfo();

  typedef struct FileHeader {
    // BitMap File Header
    char identifier[2];
    uint32_t fileSize;
    uint16_t reserved1; // must be zero
    uint16_t reserved2; // must be zero
    uint32_t bitmapDataOffset; // offset of BitMap data in bytes
  } FileHeader;

  typedef struct InfoHeader {
    // BitMap Info Header
    uint32_t bitmapHeaderSize; // size of BitMap Info Header in bytes
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t bitmapDataSize;
    uint32_t hResolution;
    uint32_t vResolution;
    uint32_t colors;
    uint32_t importantColors;
  } InfoHeader;
  
  FileHeader fileHeader;
  InfoHeader infoHeader;
  char *data, *palette;

  inline int getHeaderSize() {
    return FILEHEADER_SIZE+INFOHEADER_SIZE;
  };

  void writeToFile(FILE*);
  void writePaletteGray(FILE*);
  void writePaletteRedToRed(FILE*);
  void writePaletteBlueToRed(FILE*);

  void readGrayscales(FILE*, std::unique_ptr<char>& grays);
  void readColors(FILE*, std::unique_ptr<char>& reds, std::unique_ptr<char>& greens, std::unique_ptr<char>& blues);
  void readReds(FILE*, std::unique_ptr<char>& reds);
  void readGreens(FILE*, std::unique_ptr<char>& greens);
  void readBlues(FILE*, std::unique_ptr<char>& blues);

  // Visual System Methods
  void readLuminences(FILE*, std::unique_ptr<char>& luminences);
  void readRedGreenOpponents(FILE*, std::unique_ptr<char>& regGreenOpponents);
  void readYellowBlueOpponents(FILE*, std::unique_ptr<char>& yellowBlueOpponents);

  void hsv2rgb(float h, float s, float v, float& r,
	       float& g, float& b) const;
  
  
  template<typename T>
  void writeDataElement(FILE*, T& elem);

  template<typename T>
  void readDataElement(FILE*, T& elem);
};


#endif
