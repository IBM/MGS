/*
=================================================================
Licensed Materials - Property of IBM

"Restricted Materials of IBM"

BMC-YKT-03-25-2018

(C) Copyright IBM Corp. 2005-2017  All rights reserved

US Government Users Restricted Rights -
Use, duplication or disclosure restricted by
GSA ADP Schedule Contract with IBM Corp.

================================================================
*/
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
