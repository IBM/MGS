#https://medium.com/mobileforgood/patterns-for-continuous-integration-with-docker-on-travis-ci-71857fff14c5
## Set proxy server, replace host:port with values for your servers
## if you're behind a proxy server to get to Internet
#ENV http_proxy host:port
#ENV https_proxy host:port

# Use an official Python runtime as a parent image
# FROM python:2.7-slim
FROM python:3.6-slim

# Set the working directory to /app
#WORKDIR /app

# Copy the current directory contents into the container at /app
#ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
#EXPOSE 80

# Define environment variable
#ENV NAME World

# Run app.py when the container launches
#CMD ["python", "app.py"]

FROM centos:centos6

RUN yum groupinstall -y 'Development Tools'
RUN yum install -y openmpi openmpi-devel zlib-devel texinfo gstreamer-plugins-base-devel libXext-devel libGLU-devel libXt-devel libXrender-devel libXinerama-devel libpng-devel libXrandr-devel libXi-devel libXft-devel libjpeg-turbo-devel libXcursor-devel readline-devel ncurses-devel python python-devel wget tar
RUN wget http://sourceforge.net/projects/flex/files/flex/2.5.4.a/flex-2.5.4a.tar.bz2
RUN tar -xf flex-2.5.4a.tar.bz2; cd flex-2.5.4; ./configure; make install; make
ENV PATH=/usr/lib64/openmpi/bin/:$PATH

RUN wget --no-check-certificate https://pypi.python.org/packages/source/s/setuptools/setuptools-1.4.2.tar.gz
RUN tar -xvf setuptools-1.4.2.tar.gz; cd setuptools-1.4.2; python setup.py install
RUN curl https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py | python -
RUN pip install numpy

ADD . /nts
RUN cd /nts; source ./setenv_nts; ./make_nts LINUX
RUN useradd nts
USER nts
