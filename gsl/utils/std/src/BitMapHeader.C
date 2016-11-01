#include "BitMapHeader.h"
#include <sstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INDEX(x, y) (((y) * (infoHeader.width)) + (x))

#define BIGENDIAN

BitMapHeader::BitMapHeader() {
  fileHeader.identifier[0] = 'B';
  fileHeader.identifier[1] = 'M';
  fileHeader.fileSize = FILEHEADER_SIZE + INFOHEADER_SIZE;
  fileHeader.reserved1 = 0;
  fileHeader.reserved2 = 0;
  fileHeader.bitmapDataOffset =
      FILEHEADER_SIZE + INFOHEADER_SIZE + PALETTE_SIZE;

  infoHeader.bitmapHeaderSize = INFOHEADER_SIZE;
  infoHeader.width = 0;
  infoHeader.height = 0;
  infoHeader.planes = 1;
  infoHeader.bitsPerPixel = 8;
  infoHeader.compression = 0;
  infoHeader.bitmapDataSize = 0;
  infoHeader.hResolution = 0;
  infoHeader.vResolution = 0;
  infoHeader.colors = 0;  // 256;
  infoHeader.importantColors = 0;

  data = 0;
  palette = new char[PALETTE_SIZE];
}

BitMapHeader::BitMapHeader(char* id, uint32_t fsz, uint16_t rsvd1,
                           uint16_t rsvd2, uint32_t bmDataOffset,
                           uint32_t bmHeaderSize, uint32_t w, uint32_t h,
                           uint16_t p, uint16_t bpp, uint32_t c,
                           uint32_t bmDataSize, uint32_t hRes, uint32_t vRes,
                           uint32_t clrs, uint32_t impClrs) {
  std::copy(id, id + FILEHEADER_ID_LENGTH, fileHeader.identifier);
  // strcpy(fileHeader.identifier, id);
  fileHeader.fileSize = fsz;
  fileHeader.reserved1 = rsvd1;
  fileHeader.reserved2 = rsvd2;
  fileHeader.bitmapDataOffset = bmDataOffset;

  infoHeader.bitmapHeaderSize = bmHeaderSize;
  infoHeader.width = w;
  infoHeader.height = h;
  infoHeader.planes = p;
  infoHeader.bitsPerPixel = bpp;
  infoHeader.compression = c;
  infoHeader.bitmapDataSize = bmDataSize;
  infoHeader.hResolution = hRes;
  infoHeader.vResolution = vRes;
  infoHeader.colors = clrs;
  infoHeader.importantColors = impClrs;

  data = 0;
}

std::string BitMapHeader::getHeaderInfo() {
  std::ostringstream os;
  os << "Contents of BMP header:\n\n"
     << "FILETYPE =\t\t" << fileHeader.identifier << "\n"
     << "FILESIZE =\t\t" << fileHeader.fileSize << "\n"
     << "RESERVED1 =\t\t" << fileHeader.reserved1 << "\n"
     << "RESERVED2 =\t\t" << fileHeader.reserved2 << "\n"
     << "BITMAPOFFSET =\t\t" << fileHeader.bitmapDataOffset << "\n"
     << "HEADERSIZE =\t\t" << infoHeader.bitmapHeaderSize << "\n"
     << "WIDTH =\t\t" << infoHeader.width << "\n"
     << "HEIGHT =\t\t" << infoHeader.height << "\n"
     << "PLANES =\t\t" << infoHeader.planes << "\n"
     << "BITSPERPIXEL =\t\t" << infoHeader.bitsPerPixel << "\n"
     << "COMPRESSION =\t\t" << infoHeader.compression << "\n"
     << "BITMAPDATASIZE =\t\t" << infoHeader.bitmapDataSize << "\n"
     << "HORIZRESOLUTION =\t\t" << infoHeader.hResolution << "\n"
     << "VERTRESOLUTION =\t\t" << infoHeader.vResolution << "\n"
     << "COLORSUSED =\t\t" << infoHeader.colors << "\n"
     << "IMPORTANTCOLORS =\t\t" << infoHeader.importantColors << "\n";
  return os.str();
}

void BitMapHeader::writeToFile(FILE* out) {

  assert(infoHeader.bitsPerPixel == 8);

  infoHeader.bitmapDataSize = infoHeader.width * infoHeader.height;

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  fileHeader.fileSize = FILEHEADER_SIZE + INFOHEADER_SIZE + PALETTE_SIZE +
                        ((infoHeader.width + rowPadding) * infoHeader.height);

  fwrite(fileHeader.identifier, sizeof(char), 2, out);
  writeDataElement(out, fileHeader.fileSize);
  writeDataElement(out, fileHeader.reserved1);
  writeDataElement(out, fileHeader.reserved2);
  writeDataElement(out, fileHeader.bitmapDataOffset);

  writeDataElement(out, infoHeader.bitmapHeaderSize);
  writeDataElement(out, infoHeader.width);
  writeDataElement(out, infoHeader.height);
  writeDataElement(out, infoHeader.planes);
  writeDataElement(out, infoHeader.bitsPerPixel);
  writeDataElement(out, infoHeader.compression);
  writeDataElement(out, infoHeader.bitmapDataSize);
  writeDataElement(out, infoHeader.hResolution);
  writeDataElement(out, infoHeader.vResolution);
  writeDataElement(out, infoHeader.colors);
  writeDataElement(out, infoHeader.importantColors);

  writePaletteGray(out);
  //  writePaletteRedToRed(out);
  //  writePaletteBlueToRed(out);

  if (data) {
    char zero = 0;
    for (unsigned int y = 0; y < infoHeader.height; ++y) {
      for (unsigned int x = 0; x < infoHeader.width; ++x) {
        fwrite(&data[INDEX(x, y)], sizeof(char), 1, out);
      }
      for (unsigned int i = 0; i < rowPadding; ++i) {
        fwrite(&zero, sizeof(char), 1, out);
      }
    }
  }
}

void BitMapHeader::readGrayscales(FILE* in, std::auto_ptr<char>& grays) {
  size_t s = fread(fileHeader.identifier, sizeof(char), 2, in);
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
  readDataElement(in, fileHeader.fileSize);
  readDataElement(in, fileHeader.reserved1);
  readDataElement(in, fileHeader.reserved2);
  readDataElement(in, fileHeader.bitmapDataOffset);

  readDataElement(in, infoHeader.bitmapHeaderSize);
  readDataElement(in, infoHeader.width);
  readDataElement(in, infoHeader.height);
  readDataElement(in, infoHeader.planes);
  readDataElement(in, infoHeader.bitsPerPixel);
  readDataElement(in, infoHeader.compression);
  readDataElement(in, infoHeader.bitmapDataSize);
  readDataElement(in, infoHeader.hResolution);
  readDataElement(in, infoHeader.vResolution);
  readDataElement(in, infoHeader.colors);
  readDataElement(in, infoHeader.importantColors);

  assert(fileHeader.identifier[0] == 'B');
  assert(fileHeader.identifier[1] == 'M');
  assert(infoHeader.bitsPerPixel == 8);

  char* g = new char[infoHeader.height * infoHeader.width];

  char c;
  int d = 0;
  // check for standard grayscale palette
  bool grayscale = true;
  for (unsigned int i = 0; i < PALETTE_SIZE; ++i) {
    s = fread(&c, sizeof(char), 1, in);
    palette[i] = c;
    if (i % 4 == 3) {
      assert(c == 0);
      ++d;
    } else if (c != d) {
      grayscale = false;
    }
  }

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  for (unsigned int y = 0; y < infoHeader.height; ++y) {
    for (unsigned int x = 0; x < infoHeader.width; ++x) {
      if (grayscale)
        s = fread(&g[INDEX(x, y)], sizeof(char), 1, in);
      else {
        s = fread(&c, sizeof(char), 1, in);
        g[INDEX(x, y)] = char(0.3 * float(palette[c * 4]) +
                              0.59 * float(palette[c * 4 + 1]) +
                              0.11 * float(palette[c * 4 + 2]));
      }
    }
    for (unsigned int i = 0; i < rowPadding; ++i) {
      s = fread(&c, sizeof(char), 1, in);
    }
  }
  grays.reset(g);
}

void BitMapHeader::readColors(FILE* in, std::auto_ptr<char>& reds,
                              std::auto_ptr<char>& greens,
                              std::auto_ptr<char>& blues) {
  size_t s = fread(fileHeader.identifier, sizeof(char), 2, in);
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
  readDataElement(in, fileHeader.fileSize);
  readDataElement(in, fileHeader.reserved1);
  readDataElement(in, fileHeader.reserved2);
  readDataElement(in, fileHeader.bitmapDataOffset);

  readDataElement(in, infoHeader.bitmapHeaderSize);
  readDataElement(in, infoHeader.width);
  readDataElement(in, infoHeader.height);
  readDataElement(in, infoHeader.planes);
  readDataElement(in, infoHeader.bitsPerPixel);
  readDataElement(in, infoHeader.compression);
  readDataElement(in, infoHeader.bitmapDataSize);
  readDataElement(in, infoHeader.hResolution);
  readDataElement(in, infoHeader.vResolution);
  readDataElement(in, infoHeader.colors);
  readDataElement(in, infoHeader.importantColors);

  assert(fileHeader.identifier[0] == 'B');
  assert(fileHeader.identifier[1] == 'M');
  assert(infoHeader.bitsPerPixel == 8);

  char c;
  for (unsigned int i = 0; i < PALETTE_SIZE; ++i) {
    s = fread(&c, sizeof(char), 1, in);
    palette[i] = c;
    if (i % 4 == 3) assert(c == 0);
  }

  char* r = new char[infoHeader.height * infoHeader.width];
  char* g = new char[infoHeader.height * infoHeader.width];
  char* b = new char[infoHeader.height * infoHeader.width];

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  for (unsigned int y = 0; y < infoHeader.height; ++y) {
    for (unsigned int x = 0; x < infoHeader.width; ++x) {
      s = fread(&c, sizeof(char), 1, in);
      r[INDEX(x, y)] = palette[c * 4];
      g[INDEX(x, y)] = palette[c * 4 + 1];
      b[INDEX(x, y)] = palette[c * 4 + 2];
    }
    for (unsigned int i = 0; i < rowPadding; ++i) {
      s = fread(&c, sizeof(char), 1, in);
    }
  }
  reds.reset(r);
  greens.reset(g);
  blues.reset(b);
}

void BitMapHeader::readReds(FILE* in, std::auto_ptr<char>& reds) {
  size_t s = fread(fileHeader.identifier, sizeof(char), 2, in);
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
  readDataElement(in, fileHeader.fileSize);
  readDataElement(in, fileHeader.reserved1);
  readDataElement(in, fileHeader.reserved2);
  readDataElement(in, fileHeader.bitmapDataOffset);

  readDataElement(in, infoHeader.bitmapHeaderSize);
  readDataElement(in, infoHeader.width);
  readDataElement(in, infoHeader.height);
  readDataElement(in, infoHeader.planes);
  readDataElement(in, infoHeader.bitsPerPixel);
  readDataElement(in, infoHeader.compression);
  readDataElement(in, infoHeader.bitmapDataSize);
  readDataElement(in, infoHeader.hResolution);
  readDataElement(in, infoHeader.vResolution);
  readDataElement(in, infoHeader.colors);
  readDataElement(in, infoHeader.importantColors);

  assert(fileHeader.identifier[0] == 'B');
  assert(fileHeader.identifier[1] == 'M');
  assert(infoHeader.bitsPerPixel == 8);

  char c;
  for (unsigned int i = 0; i < PALETTE_SIZE; ++i) {
    s = fread(&c, sizeof(char), 1, in);
    palette[i] = c;
    if (i % 4 == 3) assert(c == 0);
  }

  char* r = new char[infoHeader.height * infoHeader.width];

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  for (unsigned int y = 0; y < infoHeader.height; ++y) {
    for (unsigned int x = 0; x < infoHeader.width; ++x) {
      s = fread(&c, sizeof(char), 1, in);
      r[INDEX(x, y)] = palette[c * 4];
    }
    for (unsigned int i = 0; i < rowPadding; ++i) {
      s = fread(&c, sizeof(char), 1, in);
    }
  }
  reds.reset(r);
}

void BitMapHeader::readGreens(FILE* in, std::auto_ptr<char>& greens) {
  size_t s = fread(fileHeader.identifier, sizeof(char), 2, in);
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
  readDataElement(in, fileHeader.fileSize);
  readDataElement(in, fileHeader.reserved1);
  readDataElement(in, fileHeader.reserved2);
  readDataElement(in, fileHeader.bitmapDataOffset);

  readDataElement(in, infoHeader.bitmapHeaderSize);
  readDataElement(in, infoHeader.width);
  readDataElement(in, infoHeader.height);
  readDataElement(in, infoHeader.planes);
  readDataElement(in, infoHeader.bitsPerPixel);
  readDataElement(in, infoHeader.compression);
  readDataElement(in, infoHeader.bitmapDataSize);
  readDataElement(in, infoHeader.hResolution);
  readDataElement(in, infoHeader.vResolution);
  readDataElement(in, infoHeader.colors);
  readDataElement(in, infoHeader.importantColors);

  assert(fileHeader.identifier[0] == 'B');
  assert(fileHeader.identifier[1] == 'M');
  assert(infoHeader.bitsPerPixel == 8);

  char c;
  for (unsigned int i = 0; i < PALETTE_SIZE; ++i) {
    s = fread(&c, sizeof(char), 1, in);
    palette[i] = c;
    if (i % 4 == 3) assert(c == 0);
  }

  char* g = new char[infoHeader.height * infoHeader.width];

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  for (unsigned int y = 0; y < infoHeader.height; ++y) {
    for (unsigned int x = 0; x < infoHeader.width; ++x) {
      s = fread(&c, sizeof(char), 1, in);
      g[INDEX(x, y)] = palette[c * 4 + 1];
    }
    for (unsigned int i = 0; i < rowPadding; ++i) {
      s = fread(&c, sizeof(char), 1, in);
    }
  }
  greens.reset(g);
}

void BitMapHeader::readBlues(FILE* in, std::auto_ptr<char>& blues) {
  size_t s = fread(fileHeader.identifier, sizeof(char), 2, in);
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
  readDataElement(in, fileHeader.fileSize);
  readDataElement(in, fileHeader.reserved1);
  readDataElement(in, fileHeader.reserved2);
  readDataElement(in, fileHeader.bitmapDataOffset);

  readDataElement(in, infoHeader.bitmapHeaderSize);
  readDataElement(in, infoHeader.width);
  readDataElement(in, infoHeader.height);
  readDataElement(in, infoHeader.planes);
  readDataElement(in, infoHeader.bitsPerPixel);
  readDataElement(in, infoHeader.compression);
  readDataElement(in, infoHeader.bitmapDataSize);
  readDataElement(in, infoHeader.hResolution);
  readDataElement(in, infoHeader.vResolution);
  readDataElement(in, infoHeader.colors);
  readDataElement(in, infoHeader.importantColors);

  assert(fileHeader.identifier[0] == 'B');
  assert(fileHeader.identifier[1] == 'M');
  assert(infoHeader.bitsPerPixel == 8);

  char c;
  for (unsigned int i = 0; i < PALETTE_SIZE; ++i) {
    s = fread(&c, sizeof(char), 1, in);
    palette[i] = c;
    if (i % 4 == 3) assert(c == 0);
  }

  char* b = new char[infoHeader.height * infoHeader.width];

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  for (unsigned int y = 0; y < infoHeader.height; ++y) {
    for (unsigned int x = 0; x < infoHeader.width; ++x) {
      s = fread(&c, sizeof(char), 1, in);
      b[INDEX(x, y)] = palette[c * 4 + 2];
    }
    for (unsigned int i = 0; i < rowPadding; ++i) {
      s = fread(&c, sizeof(char), 1, in);
    }
  }
  blues.reset(b);
}

void BitMapHeader::readLuminences(FILE* in, std::auto_ptr<char>& luminences) {
  size_t s = fread(fileHeader.identifier, sizeof(char), 2, in);
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
  readDataElement(in, fileHeader.fileSize);
  readDataElement(in, fileHeader.reserved1);
  readDataElement(in, fileHeader.reserved2);
  readDataElement(in, fileHeader.bitmapDataOffset);

  readDataElement(in, infoHeader.bitmapHeaderSize);
  readDataElement(in, infoHeader.width);
  readDataElement(in, infoHeader.height);
  readDataElement(in, infoHeader.planes);
  readDataElement(in, infoHeader.bitsPerPixel);
  readDataElement(in, infoHeader.compression);
  readDataElement(in, infoHeader.bitmapDataSize);
  readDataElement(in, infoHeader.hResolution);
  readDataElement(in, infoHeader.vResolution);
  readDataElement(in, infoHeader.colors);
  readDataElement(in, infoHeader.importantColors);

  assert(fileHeader.identifier[0] == 'B');
  assert(fileHeader.identifier[1] == 'M');
  assert(infoHeader.bitsPerPixel == 8);

  char c;
  for (unsigned int i = 0; i < PALETTE_SIZE; ++i) {
    s = fread(&c, sizeof(char), 1, in);
    palette[i] = c;
    if (i % 4 == 3) assert(c == 0);
  }

  char* l = new char[infoHeader.height * infoHeader.width];

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  for (unsigned int y = 0; y < infoHeader.height; ++y) {
    for (unsigned int x = 0; x < infoHeader.width; ++x) {
      s = fread(&c, sizeof(char), 1, in);
      l[INDEX(x, y)] = palette[c * 4] + palette[c * 4 + 1];  // R+G
    }
    for (unsigned int i = 0; i < rowPadding; ++i) {
      s = fread(&c, sizeof(char), 1, in);
    }
  }
  luminences.reset(l);
}

void BitMapHeader::readRedGreenOpponents(
    FILE* in, std::auto_ptr<char>& redGreenOpponents) {
  size_t s = fread(fileHeader.identifier, sizeof(char), 2, in);
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
  readDataElement(in, fileHeader.fileSize);
  readDataElement(in, fileHeader.reserved1);
  readDataElement(in, fileHeader.reserved2);
  readDataElement(in, fileHeader.bitmapDataOffset);

  readDataElement(in, infoHeader.bitmapHeaderSize);
  readDataElement(in, infoHeader.width);
  readDataElement(in, infoHeader.height);
  readDataElement(in, infoHeader.planes);
  readDataElement(in, infoHeader.bitsPerPixel);
  readDataElement(in, infoHeader.compression);
  readDataElement(in, infoHeader.bitmapDataSize);
  readDataElement(in, infoHeader.hResolution);
  readDataElement(in, infoHeader.vResolution);
  readDataElement(in, infoHeader.colors);
  readDataElement(in, infoHeader.importantColors);

  assert(fileHeader.identifier[0] == 'B');
  assert(fileHeader.identifier[1] == 'M');
  assert(infoHeader.bitsPerPixel == 8);

  char c;
  for (unsigned int i = 0; i < PALETTE_SIZE; ++i) {
    s = fread(&c, sizeof(char), 1, in);
    palette[i] = c;
    if (i % 4 == 3) assert(c == 0);
  }

  char* b = new char[infoHeader.height * infoHeader.width];

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  for (unsigned int y = 0; y < infoHeader.height; ++y) {
    for (unsigned int x = 0; x < infoHeader.width; ++x) {
      s = fread(&c, sizeof(char), 1, in);
      b[INDEX(x, y)] = palette[c * 4] - palette[c * 4 + 1];
    }
    for (unsigned int i = 0; i < rowPadding; ++i) {
      s = fread(&c, sizeof(char), 1, in);
    }
  }
  redGreenOpponents.reset(b);
}

void BitMapHeader::readYellowBlueOpponents(
    FILE* in, std::auto_ptr<char>& yellowBlueOpponents) {
  size_t s = fread(fileHeader.identifier, sizeof(char), 2, in);
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
  readDataElement(in, fileHeader.fileSize);
  readDataElement(in, fileHeader.reserved1);
  readDataElement(in, fileHeader.reserved2);
  readDataElement(in, fileHeader.bitmapDataOffset);

  readDataElement(in, infoHeader.bitmapHeaderSize);
  readDataElement(in, infoHeader.width);
  readDataElement(in, infoHeader.height);
  readDataElement(in, infoHeader.planes);
  readDataElement(in, infoHeader.bitsPerPixel);
  readDataElement(in, infoHeader.compression);
  readDataElement(in, infoHeader.bitmapDataSize);
  readDataElement(in, infoHeader.hResolution);
  readDataElement(in, infoHeader.vResolution);
  readDataElement(in, infoHeader.colors);
  readDataElement(in, infoHeader.importantColors);

  assert(fileHeader.identifier[0] == 'B');
  assert(fileHeader.identifier[1] == 'M');
  assert(infoHeader.bitsPerPixel == 8);

  char c;
  for (unsigned int i = 0; i < PALETTE_SIZE; ++i) {
    s = fread(&c, sizeof(char), 1, in);
    palette[i] = c;
    if (i % 4 == 3) assert(c == 0);
  }

  char* b = new char[infoHeader.height * infoHeader.width];

  unsigned rowPadding = (4 - (infoHeader.width % 4)) % 4;
  for (unsigned int y = 0; y < infoHeader.height; ++y) {
    for (unsigned int x = 0; x < infoHeader.width; ++x) {
      s = fread(&c, sizeof(char), 1, in);
      b[INDEX(x, y)] = palette[c * 4] + palette[c * 4 + 1] - palette[c * 4 + 2];
    }
    for (unsigned int i = 0; i < rowPadding; ++i) {
      s = fread(&c, sizeof(char), 1, in);
    }
  }
  yellowBlueOpponents.reset(b);
}

template <typename T>
void BitMapHeader::writeDataElement(FILE* out, T& elem) {
#ifdef BIGENDIAN
  int size = sizeof(T);
  for (int i = size - 1; i >= 0; --i) {
    fwrite(((char*)&elem) + i, sizeof(char), 1, out);
  }
#else
  fwrite(&elem, sizeof(elem), 1, out);
#endif
}

template <typename T>
void BitMapHeader::readDataElement(FILE* in, T& elem) {
#ifdef BIGENDIAN
  int size = sizeof(T);
  for (int i = size - 1; i >= 0; --i) {
    size_t s = fread(((char*)&elem) + i, sizeof(char), 1, in);
  }
#else
  s = fread(&elem, sizeof(elem), 1, in);
#endif
  if (feof(in)) {
    std::cerr << "Unexpected EOF in BitMapHeader!" << std::endl;
    exit(-1);
  }
  if (ferror(in)) {
    std::cerr << "Unexpected error in BitMapHeader!" << std::endl;
    exit(-1);
  }
}

void BitMapHeader::writePaletteGray(FILE* out) {
  char array[4];
  unsigned char i = 0;
  while (true) {
    std::fill(array, array + 4, i);
    // memset(array, i, 4);
    fwrite(array, sizeof(array), 1, out);
    if (i == 255) break;
    ++i;
  }
}

void BitMapHeader::writePaletteRedToRed(FILE* out) {

  float h, r, g, b;
  unsigned char rc, gc, bc;
  char zero = 0;
  for (unsigned i = 0; i < 256; ++i) {
    h = i;
    h = (h / 255) * 360;
    hsv2rgb(h, 1, 1, r, g, b);

    rc = char(r * 255.0);
    gc = char(g * 255.0);
    bc = char(b * 255.0);

    //       fwrite(&rc, sizeof(char), 1, out);
    //       fwrite(&gc, sizeof(char), 1, out);
    //       fwrite(&bc, sizeof(char), 1, out);

    fwrite(&bc, sizeof(char), 1, out);
    fwrite(&gc, sizeof(char), 1, out);
    fwrite(&rc, sizeof(char), 1, out);

    fwrite(&zero, sizeof(char), 1, out);
  }
}

void BitMapHeader::writePaletteBlueToRed(FILE* out) {

  float h, r, g, b;
  unsigned char rc, gc, bc;
  char zero = 0;
  for (unsigned i = 0; i < 256; ++i) {
    h = i;
    h = (h / 255) * 250;
    hsv2rgb(h, 1, 1, r, g, b);

    rc = char(r * 255.0);
    gc = char(g * 255.0);
    bc = char(b * 255.0);

    fwrite(&rc, sizeof(char), 1, out);
    fwrite(&gc, sizeof(char), 1, out);
    fwrite(&bc, sizeof(char), 1, out);

    //        fwrite(&bc, sizeof(char), 1, out);
    //        fwrite(&gc, sizeof(char), 1, out);
    //        fwrite(&rc, sizeof(char), 1, out);

    fwrite(&zero, sizeof(char), 1, out);
  }
}

void BitMapHeader::hsv2rgb(float h, float s, float v, float& r, float& g,
                           float& b) const {
  float i, f, p, q, t;

  if ((h > 360) || (s > 1) || (v > 1) || (h < 0) || (s < 0) || (v < 0)) {
    printf("Error in color_conversions.c: expecting hsv to be in [0,1]\n");
    printf("      Instead, got h = %g, s = %g, v = %g\n", h, s, v);
    exit(1);
  }

  if (s == 0) {
    r = v;
    g = v;
    b = v;
    return;
  }
  if (h == 360) h = 0;

  h = (float)h / 60.0; /* h is now in the range [0,6)  */

  i = floor(h); /* largest interger <= h        */

  f = h - i; /* fractional part of h         */

  p = v * (1 - s);
  q = v * (1 - (s * f));
  t = v * (1 - (s * (1 - f)));
  if (i == 0) {
    r = v;
    g = t;
    b = p;
  } else if (i == 1) {
    r = q;
    g = v;
    b = p;
  } else if (i == 2) {
    r = p;
    g = v;
    b = t;
  } else if (i == 3) {
    r = p;
    g = q;
    b = v;
  } else if (i == 4) {
    r = t;
    g = p;
    b = v;
  } else if (i == 5) {
    r = v;
    g = p;
    b = q;
  }
}
