// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
// 
// =============================================================================
#ifndef __FILEFOLDERUTILS_H__
#define __FILEFOLDERUTILS_H__

#include <string>
#include <vector>
#include <map>

class FileFolderUtils
{
  public:
  static bool isFolderExist(const std::string& str,
                            const std::string& fileExtension = ".dat,.txt");
  static bool isFileExist(const std::string& file_name);

  static std::map< std::string, std::vector<float> > readCsvFile(const std::string& file_name, unsigned int & num_col);
};

#endif
