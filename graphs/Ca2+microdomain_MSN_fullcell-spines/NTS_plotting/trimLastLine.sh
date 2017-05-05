#!/bin/bash
ls $1/*dat* | xargs -I{} sed -i '$ d' {}
