# Datathon

Hello!  This is me, Sarah Hewitt, PhD student at the University of Southampton, adding my work to this repository.

I use the Anaconda suite to write my code, and I've included two files.  One is written for Jupyter, and uses Pandas to open a month of taxi trip data, and drop the columns of data we didn't need.  It also includes some code to gather a 1% sample from a month of data, which is then pulled into the Orange app.  If you use this, and open the ows file, you will need to have the Geo add-on installed.  This was dead easy if you're using an Apple device, not so with Windows.

If you're using a Windows device, you may need to install some extra files:

pip install --upgrade setuptools
pip install ez_setup

Then you need:
conda install shapely
conda install geos

