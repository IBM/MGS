FROM centos:centos6

RUN yum groupinstall -y 'Development Tools'
RUN yum install -y openmpi openmpi-devel zlib-devel texinfo gstreamer-plugins-base-devel libXext-devel libGLU-devel libXt-devel libXrender-devel libXinerama-devel libpng-devel libXrandr-devel libXi-devel libXft-devel libjpeg-turbo-devel libXcursor-devel readline-devel ncurses-devel python python-devel wget tar
RUN wget http://sourceforge.net/projects/flex/files/flex/2.5.4.a/flex-2.5.4a.tar.bz2
RUN tar -xf flex-2.5.4a.tar.bz2; cd flex-2.5.4; ./configure; make install; make
ENV PATH=/usr/lib64/openmpi/bin/:$PATH

ADD . /nts

RUN cd /nts; source ./setenv_nts; ./make_nts LINUX

RUN wget --no-check-certificate https://pypi.python.org/packages/source/s/setuptools/setuptools-1.4.2.tar.gz
RUN tar -xvf setuptools-1.4.2.tar.gz; cd setuptools-1.4.2; python setup.py install
RUN curl https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py | python -
RUN pip install numpy

RUN useradd nts
USER nts
